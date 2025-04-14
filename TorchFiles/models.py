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
        #person_conf = person_conf/person_conf.max()
        return cxcywh, person_conf

class yolo_out_splitter_without_score_scaling(torch.nn.Module):
    def __init__(self, score_threshold=0.0):
        super(yolo_out_splitter_without_score_scaling, self).__init__()
        self.score_threshold = score_threshold

    def forward(self, yolo_raw_out):
        # yolo_raw_out shape: [1,84,8400]
        yolo_raw_out = yolo_raw_out.permute(0, 2, 1).squeeze(0) # [8400,84]
        yolo_raw_out = yolo_raw_out[yolo_raw_out[..., 4] > self.score_threshold]
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
        cxcywh = cxcywh # [300,4] scaled 0-640
        person_conf = rtdetr_raw_out[..., 4] # [300]
        person_conf = person_conf/person_conf.max()
        return cxcywh, person_conf

class rtdetr_out_splitter_without_score_scaling(torch.nn.Module):
    def __init__(self, score_threshold=0.0):
        super(rtdetr_out_splitter_without_score_scaling, self).__init__()
        self.score_threshold = score_threshold

    def forward(self, rtdetr_raw_out):
        # rtdetr_raw_out shape: [1,300,84]
        rtdetr_raw_out = rtdetr_raw_out.squeeze(0) # [300,84]
        rtdetr_raw_out = rtdetr_raw_out[rtdetr_raw_out[..., 4] > self.score_threshold]
        cxcywh = rtdetr_raw_out[..., :4] # [300,4] but sclaed 0-1
        cxcywh = cxcywh*640 # [300,4] scaled 0-640
        person_conf = rtdetr_raw_out[..., 4] # [300]
        return cxcywh, person_conf

class cxcywh2xyxy(torch.nn.Module):
    def __init__(self):
        super(cxcywh2xyxy, self).__init__()

    def forward(self, cxcywh):
        # cxcywh shape: [N,4]
        xyxy = torchvision.ops.box_convert(cxcywh, 'cxcywh', 'xyxy')
        return xyxy

class NMS(torch.nn.Module):
    def __init__(self, score_threshold=0.5, iou_threshold=0.5):
        super(NMS, self).__init__()
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def forward(self, xyxy, person_conf):
        # xyxy shape: [N,4]
        # person_conf shape: [N]
        boxes_and_scores = torch.cat((xyxy, person_conf.unsqueeze(1)), dim=1) # [M,5]
        boxes_and_scores = boxes_and_scores[boxes_and_scores[:,4] >= self.score_threshold]
        selected_indices = torchvision.ops.nms(boxes_and_scores[:, :4], boxes_and_scores[:, 4], iou_threshold=self.iou_threshold) # [N]
        boxes_and_scores = boxes_and_scores[selected_indices]
        return boxes_and_scores

class NMS_without_score_thresholding(torch.nn.Module):
    def __init__(self, iou_threshold=0.5):
        super(NMS_without_score_thresholding, self).__init__()
        self.iou_threshold = iou_threshold

    def forward(self, xyxy, person_conf):
        # xyxy shape: [N,4]
        # person_conf shape: [N]
        boxes_and_scores = torch.cat((xyxy, person_conf.unsqueeze(1)), dim=1) # [M,5]
        selected_indices = torchvision.ops.nms(boxes_and_scores[:, :4], boxes_and_scores[:, 4], iou_threshold=self.iou_threshold) # [N]
        boxes_and_scores = boxes_and_scores[selected_indices]
        return boxes_and_scores

class image_sender(torch.nn.Module):
    def __init__(self):
        super(image_sender, self).__init__()

    def forward(self, image):
        # image shape: [1,3,640,640]
        return image, image
    
class RTDETR_postprocess(torch.nn.Module):
    def __init__(self, score_threshold=0.5, iou_threshold=0.5):
        super(RTDETR_postprocess, self).__init__()
        self.rtdetr_out_splitter = rtdetr_out_splitter()
        self.cxcywh2xyxy = cxcywh2xyxy()
        self.NMS = NMS(score_threshold, iou_threshold)

    def forward(self, rtdetr_raw_out):
        # rtdetr_raw_out shape: [1,300,84]
        cxcywh, person_conf = self.rtdetr_out_splitter(rtdetr_raw_out)
        xyxy = self.cxcywh2xyxy(cxcywh)
        boxes_and_scores = self.NMS(xyxy, person_conf)
        return boxes_and_scores

class RTDETR_postprocess_without_score_scaling(torch.nn.Module):
    def __init__(self, score_threshold=0.5, iou_threshold=0.5):
        super(RTDETR_postprocess_without_score_scaling, self).__init__()
        self.rtdetr_out_splitter = rtdetr_out_splitter_without_score_scaling()
        self.cxcywh2xyxy = cxcywh2xyxy()
        self.NMS = NMS(score_threshold, iou_threshold)

    def forward(self, rtdetr_raw_out):
        # rtdetr_raw_out shape: [1,300,84]
        cxcywh, person_conf = self.rtdetr_out_splitter(rtdetr_raw_out)
        xyxy = self.cxcywh2xyxy(cxcywh)
        boxes_and_scores = self.NMS(xyxy, person_conf)
        return boxes_and_scores

class YOLO_postprocess(torch.nn.Module):
    def __init__(self, score_threshold=0.5, iou_threshold=0.5):
        super(YOLO_postprocess, self).__init__()
        self.yolo_out_splitter = yolo_out_splitter()
        self.cxcywh2xyxy = cxcywh2xyxy()
        self.NMS = NMS(score_threshold, iou_threshold)

    def forward(self, yolo_raw_out):
        # yolo_raw_out shape: [1,84,8400]
        cxcywh, person_conf = self.yolo_out_splitter(yolo_raw_out)
        xyxy = self.cxcywh2xyxy(cxcywh)
        boxes_and_scores = self.NMS(xyxy, person_conf)
        return boxes_and_scores

class YOLO_postprocess_without_score_thresholding(torch.nn.Module):
    def __init__(self, iou_threshold=0.5):
        super(YOLO_postprocess_without_score_thresholding, self).__init__()
        self.yolo_out_splitter = yolo_out_splitter()
        self.cxcywh2xyxy = cxcywh2xyxy()
        self.NMS = NMS_without_score_thresholding(iou_threshold)

    def forward(self, yolo_raw_out):
        # yolo_raw_out shape: [1,84,8400]
        cxcywh, person_conf = self.yolo_out_splitter(yolo_raw_out)
        xyxy = self.cxcywh2xyxy(cxcywh)
        boxes_and_scores = self.NMS(xyxy, person_conf)
        return boxes_and_scores

class YOLO_postprocess_without_score_scaling(torch.nn.Module):
    def __init__(self, score_threshold=0.5, iou_threshold=0.5):
        super(YOLO_postprocess_without_score_scaling, self).__init__()
        self.yolo_out_splitter = yolo_out_splitter_without_score_scaling()
        self.cxcywh2xyxy = cxcywh2xyxy()
        self.NMS = NMS(score_threshold, iou_threshold)

    def forward(self, yolo_raw_out):
        # yolo_raw_out shape: [1,84,8400]
        cxcywh, person_conf = self.yolo_out_splitter(yolo_raw_out)
        xyxy = self.cxcywh2xyxy(cxcywh)
        boxes_and_scores = self.NMS(xyxy, person_conf)
        return boxes_and_scores

class Ensemble_postprocess(torch.nn.Module):
    def __init__(self,
                 yolo_score_threshold: float=None,
                 rtdetr_score_threshold: float=None,
                 score_threshold: float=0.5,
                 iou_threshold: float=0.5
                 ):
        super(Ensemble_postprocess, self).__init__()
        self.yolo_out_splitter = yolo_out_splitter() if yolo_score_threshold is None else yolo_out_splitter_without_score_scaling(yolo_score_threshold)
        self.rtdetr_out_splitter = rtdetr_out_splitter() if rtdetr_score_threshold is None else rtdetr_out_splitter_without_score_scaling(rtdetr_score_threshold)
        self.cxcywh2xyxy = cxcywh2xyxy()
        self.NMS = NMS(score_threshold, iou_threshold)

    def forward(self, yolo_raw_out, rtdetr_raw_out):
        # yolo_raw_out shape: [1,84,8400]
        # rtdetr_raw_out shape: [1,300,84]
        yolo_cxcywh, yolo_person_conf = self.yolo_out_splitter(yolo_raw_out)
        rtdetr_cxcywh, rtdetr_person_conf = self.rtdetr_out_splitter(rtdetr_raw_out)
        cxcywh = torch.cat((yolo_cxcywh, rtdetr_cxcywh), dim=0)
        xyxy = self.cxcywh2xyxy(cxcywh)
        person_conf = torch.cat((yolo_person_conf, rtdetr_person_conf), dim=0)
        boxes_and_scores = self.NMS(xyxy, person_conf)
        return boxes_and_scores

class Ensemble_postprocess_triple_NMS(torch.nn.Module):
    def __init__(self, yolo_score_threshold=0.5, yolo_iou_threshold=0.5, rtdetr_score_threshold=0.5, rtdetr_iou_threshold=0.5, score_threshold=0.5, iou_threshold=0.5):
        super(Ensemble_postprocess_triple_NMS, self).__init__()
        self.yolo_postprocess = YOLO_postprocess(yolo_score_threshold, yolo_iou_threshold)
        self.rtdetr_postprocess = RTDETR_postprocess(rtdetr_score_threshold, rtdetr_iou_threshold)
        self.NMS = NMS(score_threshold, iou_threshold)

    def forward(self, yolo_raw_out, rtdetr_raw_out):
        # yolo_raw_out shape: [1,84,8400]
        # rtdetr_raw_out shape: [1,300,84]
        yolo_boxes_and_scores = self.yolo_postprocess(yolo_raw_out)
        rtdetr_boxes_and_scores = self.rtdetr_postprocess(rtdetr_raw_out)
        xyxy = torch.cat((yolo_boxes_and_scores[:, :4], rtdetr_boxes_and_scores[:, :4]), dim=0)
        person_conf = torch.cat((yolo_boxes_and_scores[:, 4], rtdetr_boxes_and_scores[:, 4]), dim=0)
        boxes_and_scores = self.NMS(xyxy, person_conf)
        return boxes_and_scores