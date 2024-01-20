import numpy as np
import cv2 
import torchvision
import torch
import time
class Utils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def letterbox(im, color=(114, 114, 114), auto=True, scaleup=True, stride=32, img_size=(640, 640)):
        # print(im)
        shape = im.shape[:2]
        new_shape = img_size
        if isinstance(new_shape, int):
            new_shape=(new_shape, new_shape)
        r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
        if not scaleup:
            r = min(r, 1.0)
        
        new_unpad = int(round(shape[1]* r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw/=2
        dh/=2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
        left, right = int(round(dw-0.1)), int(round(dw+0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)
    
    @staticmethod
    def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
        """Performs Non-Maximum Suppression (NMS) on inference results
        Returns:
            detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        # print(prediction)
        nc = prediction.shape[2] - 15  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 15), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 15] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = Utils.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, landmarks, cls)
            if multi_label:
                i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 15, None], x[i, 5:15] ,j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 15:].max(1, keepdim=True)
                x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Batched NMS
            c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            #if i.shape[0] > max_det:  # limit detections
            #    i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = Utils.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded
        # print(output)
        return output
    
    @staticmethod
    def cosine_similarity(feat, tmp):
        feat_norm = np.linalg.norm(feat)
        tmp_norm = np.linalg.norm(tmp)
        if feat_norm == 0 or tmp_norm == 0:
            return 0.0
        tmp = tmp.reshape(-1, 1)
        cosine_sim = np.dot(feat, tmp)/ (feat_norm * tmp_norm)
        return cosine_sim
    
    @staticmethod
    def pytorch_cosine_similarity(feat, tmp):
        feat = torch.from_numpy(feat)
        tmp = torch.from_numpy(tmp)
        cosine_sim = torch.cosine_similarity(feat, tmp)
        return cosine_sim

    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    @staticmethod
    def crop_image(image, bbox):
        # print(bbox)
        box = bbox[:-1]
        score = bbox[-1]
        cropped_box = image[box[1]: box[3], box[0]: box[2]]
        return cropped_box

    @staticmethod
    def crop_image_feat_extraction(image, bboxes, get_images=False):
        """
        This methods take bounding images and bounding boxes
        as inputs then return list of cropped images
        """
        cropped_boxes = []
        cropped_images = []
        if bboxes is None:
            # print('No one')
            return None, None
        for bbox in bboxes:
            cropped_box=Utils.crop_image(image, bbox)
            cropped_images.append(cropped_box)
            if(cropped_box.shape[0]==0 or cropped_box.shape[1]==0):
                continue
            if(type(cropped_box) == type(None)):
                pass
            else:
                cropped_box= cv2.resize(cropped_box, (112, 112))
            cropped_box = np.transpose(cropped_box, (2, 0, 1))
            cropped_box = torch.from_numpy(cropped_box).unsqueeze(0).float()
            cropped_box.div_(255).sub_(0.5).div_(0.5)
            cropped_boxes.append(cropped_box)
        if get_images == True:
            # print(cropped_box)
            # print(cropped_images)
            return cropped_boxes, cropped_images
        return cropped_boxes
    

