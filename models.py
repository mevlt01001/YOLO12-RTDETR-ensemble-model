import torch
import torchvision

class yolo_raw_out_splitter(torch.nn.Module):
    def __init__(self):
        super(yolo_raw_out_splitter, self).__init__()

    def forward(self, yolo_raw_out):
        # yolo_raw_out shape: [1,84,8400]
        yolo_raw_out = yolo_raw_out.permute(0, 2, 1).squeeze(0) # [8400,84]
        cxcywh = yolo_raw_out[..., :4] # [8400,4]
        person_conf = yolo_raw_out[..., 4] # [8400]
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
        return cxcywh, person_conf
    
class boxes_and_scores_merger(torch.nn.Module):
    def __init__(self):
        super(boxes_and_scores_merger, self).__init__()

    def forward(self, yolo_cxcywh, yolo_person_conf, rtdetr_cxcywh, rtdetr_person_conf, ):
        # yolo_cxcywh shape: [8400,4]
        # yolo_person_conf shape: [8400]
        # rtdetr_cxcywh shape: [300,4]
        # rtdetr_person_conf shape: [300]
        cxcywh = torch.cat((yolo_cxcywh, rtdetr_cxcywh), dim=0) # [8400+300,4]
        person_conf = torch.cat((yolo_person_conf.clamp(0,1), rtdetr_person_conf.clamp(0,1)), dim=0) # [8400+300] and normalized 0-1
        xyxy = torchvision.ops.box_convert(cxcywh, 'cxcywh', 'xyxy')

        selected_indices = torchvision.ops.nms(xyxy, person_conf, iou_threshold=0.5) # [8400+300]
        xyxy = xyxy[selected_indices] # [N,4]
        person_conf = person_conf[selected_indices] # [N]

        boxes_and_scores = torch.cat((xyxy, person_conf.unsqueeze(1)), dim=1) # [N,5]
        person_conf = person_conf[person_conf>0.5]
        return boxes_and_scores



        
        