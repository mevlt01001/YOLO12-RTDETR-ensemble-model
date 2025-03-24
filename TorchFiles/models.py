import torch
import torchvision

class yolo_out_splitter(torch.nn.Module):
    def __init__(self):
        super(yolo_out_splitter, self).__init__()

    def forward(self, yolo_raw_out):
        # yolo_raw_out shape: [1,84,8400]
        yolo_raw_out = yolo_raw_out.permute(0, 2, 1).squeeze(0) # [8400,84]
        cxcywh = yolo_raw_out[..., :4] # [8400,4]
        person_conf = yolo_raw_out[..., 4] # [8400]
        person_conf = person_conf/person_conf.max()
        return cxcywh, person_conf

class rtdetr_out_splitter(torch.nn.Module):
    def __init__(self):
        super(rtdetr_out_splitter, self).__init__()

    def forward(self, rtdetr_raw_out):
        # rtdetr_raw_out shape: [1,300,84]
        rtdetr_raw_out = rtdetr_raw_out.squeeze(0) # [300,84]
        cxcywh = rtdetr_raw_out[..., :4] # [300,4] but sclaed 0-1
        cxcywh = cxcywh*640 # [300,4] scaled 0-640
        person_conf = rtdetr_raw_out[..., 4] # [300]
        person_conf = person_conf/person_conf.max()
        return cxcywh, person_conf

class cxcywh2xyxy(torch.nn.Module):
    def __init__(self):
        super(cxcywh2xyxy, self).__init__()

    def forward(self, cxcywh):
        # cxcywh shape: [N,4]
        xyxy = torchvision.ops.box_convert(cxcywh, 'cxcywh', 'xyxy')
        return xyxy
    
class NMS(torch.nn.Module):
    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, xyxy, person_conf):
        # xyxy shape: [N,4]
        # person_conf shape: [N]
        selected_indices = torchvision.ops.nms(xyxy, person_conf, iou_threshold=0.5) # [N]
        xyxy = xyxy[selected_indices] # [M,4]
        person_conf = person_conf[selected_indices] # [M]
        boxes_and_scores = torch.cat((xyxy, person_conf.unsqueeze(1)), dim=1) # [M,5]
        boxes_and_scores = boxes_and_scores[boxes_and_scores[:,4] >= 0.25]
        return boxes_and_scores

class gather_boxes_scores(torch.nn.Module):
    def __init__(self):
        super(gather_boxes_scores, self).__init__()

    def forward(self, xyxy, person_conf):
        # xyxy shape: [N,4]
        # person_conf shape: [N]
        boxes_and_scores = torch.cat((xyxy, person_conf.unsqueeze(1)), dim=1) # [N,5]
        return boxes_and_scores

class image_sender(torch.nn.Module):
    def __init__(self):
        super(image_sender, self).__init__()

    def forward(self, image):
        # image shape: [1,3,640,640]
        return image, image