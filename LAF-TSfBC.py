import os
import numpy as np
import argparse
import cv2
import pickle

def evaluate_segmentation(pred, label):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    pred_p = (flat_pred==1)
    pred_n = (flat_pred!=1)
    label_p = (flat_label==1)
    label_n = (flat_label!=1)
    tp = np.sum(pred_p * label_p)
    fp = np.sum(pred_p * label_n)
    fn = np.sum(pred_n * label_p)
    tn = np.sum(pred_n * label_n)

    return tp, fp, fn, tn


parser = argparse.ArgumentParser()
parser.add_argument('--hrgt', type=str, default="", help='Path to high recall GT.')
parser.add_argument('--hpgt', type=str, default="", help='Path to high precision GT.')
parser.add_argument('--hrpred', type=str, default="", help='Path to high recall prediction.')
parser.add_argument('--hppred', type=str, default="", help='Path to high precision prediction.')



args = parser.parse_args()

dirs = os.listdir(args.hrpred)
dirs.sort()

'''
Lfp
'''
imgs = os.listdir(args.hrgt)
mlfps = []
for d in dirs:
    dp = os.path.join(args.hrpred, d)
    fps = []
    tns = []
    for img in imgs:
        gtp = os.path.join(args.hrgt, img)
        predp = os.path.join(dp, img)
        gt = cv2.imread(gtp, -1)
        gt[gt>0]=1
        pred = cv2.imread(predp, -1)
        pred[pred>0]=1
        tp, fp, fn, tn = evaluate_segmentation(pred, gt)
        fps.append(fp)

    mfp = np.mean(fps)
    mlfps.append(mfp)


dirs = os.listdir(args.hppred)
dirs.sort()

'''
Ltp and Lfn
'''
imgs = os.listdir(args.hpgt)
mltps = []
mlfns = []
for d in dirs:
    dp = os.path.join(args.hppred, d)
    tps = []
    fns = []
    for img in imgs:
        gtp = os.path.join(args.hpgt, img)
        predp = os.path.join(dp, img)
        gt = cv2.imread(gtp, -1)
        gt[gt>0]=1
        pred = cv2.imread(predp, -1)
        pred[pred>0]=1
        tp, fp, fn, tn = evaluate_segmentation(pred, gt)
        tps.append(tp)
        fns.append(fn)
    mtp = np.mean(tps)
    mfn = np.mean(fns)
    mltps.append(mtp)
    mlfns.append(mfn)

'''
Lprecision, Lrecall, Lf1, LfIoU
'''
lprecs = []
lrecs = []
lf1s = []
lious = []
for i in range(len(mlfps)):
    mltp = mltps[i]
    mlfp = mlfps[i]
    mlfn = mlfns[i]
    lprec = mltp / (mltp + mlfp)
    lrec = mltp / (mltp + mlfn)
    lf1 = (2 * lprec * lrec) / (lprec + lrec)
    liou = mltp / (mltp+mlfp+mlfn)
    lprecs.append(lprec)
    lrecs.append(lrec)
    lf1s.append(lf1)
    lious.append(liou)

    print("ltp, lfp, lfn, ltn, lprec, lrec, lf1, liou")
    print([mltp, mlfp, mlfn, lprec, lrec, lf1, liou])

    










