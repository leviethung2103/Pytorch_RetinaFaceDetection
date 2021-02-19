# Test Face Recognizer
import argparse
import time

import cv2
import torch

from models.arcface.model_irse import IR_50
from utils.feature_extraction import extract_feature

parser = argparse.ArgumentParser(description="facerecog")
parser.add_argument("--emb_model", default="weights/backbone_ir50_asia.pth", type=str, help="dir pth model")
args = parser.parse_args()

# Face Embedding Model
face_emb = IR_50((112, 112))
face_emb.eval()
face_emb.load_state_dict(torch.load(args.emb_model))
if torch.cuda.is_available():
    print("Use CUDA")
    face_emb.cuda()
    torch.cuda.empty_cache()

# Read Image
img_path = "images/barack-obama.jpg"
image = cv2.imread(img_path)
# ! This implementation is not correct. We need to pass the image to Face Detection, then crop + align the face before getting embedding
# class 'torch.Tensor'
# torch.Size([1,512])
t1 = time.time()
emb_array = extract_feature(image, face_emb)
print("Processing Time: ", time.time() - t1)

