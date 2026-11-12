# 本地换装的内置衣服（博物馆开放馆藏，CC0）：下载原图 → 按背景色抠图（黑底的用 GrabCut）→ 抹掉领口里透出的内衬 → 写关键点标注。
# 标注坐标是抠图裁剪后（长边 1024）的像素坐标，格式见 README.md；原图换了要重新标。
# 用法: python museum.py <输出目录> [原图缓存目录]    依赖 numpy、scipy、opencv-python
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ITEMS = [
    dict(id='dragon', title='Dragon Robe（清，龙袍）', license='CC0',
         page='https://www.metmuseum.org/art/collection/search/65642',
         url='https://images.metmuseum.org/CRDImages/as/original/LC-56_210_6_sr1-001.jpg',
         thresh=18, erase_rel=[[0.455, 0, 0.545, 0.055]],
         ann={"long": True, "N": [510, 2], "SL": [325, 22], "SR": [735, 22], "AL": [328, 238], "AR": [738, 232], "HL": [150, 858], "HR": [915, 850],
              "sleeveL": [[0, 0], [326, 0], [330, 240], [0, 240]], "sleeveR": [[734, 0], [1024, 0], [1024, 236], [740, 236]],
              "ease": 1.4, "shoulder": 216, "sleeveSqueeze": 0.6}),
    dict(id='qing', title="Woman's Changfu (Informal Court Robe)（清，女常服袍）", license='CC0',
         page='https://www.artic.edu/ （经 rawpixel：https://www.rawpixel.com/image/9001266）',
         url='https://images.rawpixel.com/editor_1024/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTEyL2FpYzEwNDk1LWltYWdlLmpwZw.jpg',
         thresh=10, grabcut=True,
         ann={"long": True, "N": [345, 12], "SL": [143, 15], "SR": [550, 15], "AL": [143, 248], "AR": [550, 248], "HL": [28, 785], "HR": [650, 790],
              "sleeveL": [[0, 0], [144, 0], [144, 252], [0, 252]], "sleeveR": [[549, 0], [694, 0], [694, 252], [549, 252]],
              "ease": 1.4, "shoulder": 240, "sleeveSqueeze": 0.6}),
    dict(id='wedding', title='Cream satin wedding dress with grommet embellishments and train（婚纱）', license='CC0',
         page='https://nmaahc.si.edu/object/nmaahc_2007.3.824',
         url='https://ids.si.edu/ids/deliveryService?id=NMAAHC-2007_3_824_001',
         thresh=10, erase_rel=[[0.4, 0, 0.6, 0.025]],
         erase_poly=[[[235, 0], [420, 0], [420, 40], [380, 52], [330, 60], [280, 54], [240, 42]]],
         ann={"long": True, "N": [327, 12], "SL": [228, 45], "SR": [425, 45], "AL": [240, 150], "AR": [410, 150], "WL": [262, 205], "WR": [395, 205],
              "HL": [40, 900], "HR": [650, 900],
              "sleeveL": [[226, 25], [238, 150], [205, 195], [110, 245], [65, 215], [195, 25]],
              "sleeveR": [[427, 25], [412, 150], [440, 225], [545, 280], [570, 240], [460, 25]], "ease": 1.05}),
    dict(id='qipao', title='Red floral sleeveless qipao（旗袍）', license='CC0',
         page='https://nmaahc.si.edu/object/nmaahc_2007.3.685',
         url='https://ids.si.edu/ids/deliveryService?id=NMAAHC-2007_3_685_001',
         thresh=14,
         ann={"long": True, "N": [238, 10], "SL": [66, 58], "SR": [378, 55], "AL": [52, 215], "AR": [410, 215], "WL": [62, 390], "WR": [380, 390],
              "HL": [42, 965], "HR": [405, 965]}),
]


def cut(im, T, erase_rel=(), grabcut=False):
    """四边中位色当背景，Lab 色差 > T 算衣服；开运算、最大连通块、只填小洞（领口这类大开口留着）；可选 GrabCut 细化"""
    h, w = im.shape[:2]
    s = 1024 / max(h, w)
    im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.median(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]), axis=0)
    d = np.sqrt(((lab - bg) ** 2).sum(-1))
    m = d > T
    for x0, y0, x1, y1 in erase_rel:
        m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = False
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0

    def largest(m):
        lb, n = ndimage.label(m)
        return lb == (np.argmax(ndimage.sum(m, lb, range(1, n + 1))) + 1) if n else m
    m = largest(m)
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(holes)
    if hn:
        sz = ndimage.sum(holes, hl, range(1, hn + 1))
        m |= np.isin(hl, 1 + np.where(sz < 0.002 * h * w)[0])
    if grabcut:
        gm = np.where(m, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        gm[:4] = gm[-4:] = cv2.GC_BGD
        gm[:, :4] = gm[:, -4:] = cv2.GC_BGD
        gm[d > 3 * T] = cv2.GC_FGD
        cv2.grabCut(im, gm, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
        m = largest((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD))
    a = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)
    ys, xs = np.where(m)
    pad = 8
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h), max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    return np.dstack([im, (a * 255).astype(np.uint8)])[y0:y1, x0:x1]


def main():
    out = Path(sys.argv[1])
    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else out / 'src'
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for it in ITEMS:
        src = cache / (it['id'] + '.jpg')
        if not src.exists():
            req = urllib.request.Request(it['url'], headers={'User-Agent': 'Mozilla/5.0'})
            src.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        im = cv2.imread(str(src), cv2.IMREAD_COLOR)
        rgba = cut(im, it['thresh'], it.get('erase_rel', ()), it.get('grabcut', False))
        for poly in it.get('erase_poly', ()):
            pm = np.zeros(rgba.shape[:2], np.uint8)
            cv2.fillPoly(pm, [np.int32(poly)], 1)
            rgba[..., 3][pm > 0] = 0
        cv2.imwrite(str(out / (it['id'] + '.png')), rgba)
        (out / (it['id'] + '.json')).write_text(json.dumps(it['ann']))
        print(it['id'], rgba.shape)


if __name__ == '__main__':
    main()
