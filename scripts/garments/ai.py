# 本地换装的内置衣服（AI 生成的白底 / 灰底平铺照，原图在 ai_src/，长边 1024）：抠图 → 补回被当成背景的白衬衫 → 写关键点标注。
# 标注只写左半边（画面左边）和中线 cx，右半边按中线镜像；坐标是抠图裁剪后的像素坐标（裁剪按 T=12 的蒙版定，和标注时一致）。
# 袖子区域：long = 肩→腋下→下摆这条侧缝线外侧（长袖贴着衣身垂下）；short = 肩→腋下外侧、腋下以上；
#          poly = 写明的多边形（古装宽袖）；没有 = 无袖。
# 用法: python ai.py <输出目录>    依赖 numpy、scipy、opencv-python
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

SRC = Path(__file__).parent / 'ai_src'

# id: (cx, N, SL, AL, HL, 其它)
ITEMS = {
    'tee_white':    dict(cx=452, N=(452, 8), SL=(200, 70), AL=(185, 352), HL=(175, 840), sleeve='short'),
    'tee_black':    dict(cx=452, N=(452, 8), SL=(203, 72), AL=(193, 352), HL=(180, 840), sleeve='short'),
    'tee_stripe':   dict(cx=446, N=(446, 8), SL=(195, 75), AL=(175, 346), HL=(175, 845), sleeve='short'),
    'polo':         dict(cx=473, N=(473, 8), SL=(200, 105), AL=(185, 390), HL=(180, 880), sleeve='short'),
    'shirt_white':  dict(cx=491, N=(491, 10), SL=(283, 100), AL=(283, 385), HL=(273, 700), sleeve='long'),
    'shirt_plaid':  dict(cx=493, N=(493, 8), SL=(290, 100), AL=(292, 350), HL=(276, 690), sleeve='long'),
    'hoodie':       dict(cx=490, N=(490, 8), SL=(272, 245), AL=(268, 420), HL=(268, 750), sleeve='long'),
    'sweater':      dict(cx=475, N=(475, 8), SL=(270, 65), AL=(272, 350), HL=(285, 685), sleeve='long'),
    'cardigan':     dict(cx=485, N=(485, 10), SL=(270, 75), AL=(272, 255), HL=(270, 690), sleeve='long'),
    'denim_jacket': dict(cx=490, N=(490, 10), SL=(270, 115), AL=(265, 280), HL=(250, 690), sleeve='long'),
    'bomber':       dict(cx=502, N=(502, 8), SL=(295, 95), AL=(268, 270), HL=(300, 670), sleeve='long'),
    'down_jacket':  dict(cx=490, N=(490, 5), SL=(290, 100), AL=(282, 260), HL=(280, 600), sleeve='long'),
    'trench':       dict(cx=308, N=(308, 8), SL=(175, 75), AL=(198, 255), HL=(110, 750), WL=(200, 300), sleeve='long', long=True, ease=1.15),
    'sport':        dict(cx=500, N=(500, 8), SL=(290, 140), AL=(268, 400), HL=(285, 700), sleeve='long'),
    'suit':         dict(cx=490, N=(490, 10), SL=(270, 100), AL=(262, 375), HL=(230, 770), sleeve='long',
                         keep=[[(430, 15), (555, 15), (540, 160), (490, 210), (445, 160)]]),
    'groom':        dict(cx=472, N=(472, 5), SL=(257, 100), AL=(272, 345), HL=(222, 755), sleeve='long',
                         keep=[[(432, 0), (547, 0), (530, 120), (486, 385), (445, 120)]]),
    'jk':           dict(cx=475, N=(475, 5), SL=(268, 75), AL=(258, 380), HL=(258, 535), sleeve='long'),
    'tangzhuang':   dict(cx=510, N=(510, 8), SL=(297, 100), AL=(282, 400), HL=(252, 745), sleeve='long'),
    'dress_black':  dict(cx=277, N=(277, 8), SL=(150, 35), AL=(150, 195), HL=(5, 750), WL=(155, 325), sleeve='short', long=True, ease=1.05),
    'gown':         dict(cx=262, N=(262, 65), SL=(185, 10), AL=(198, 100), HL=(20, 860), WL=(205, 190), long=True, ease=1.05, shoulder=210),
    'qipao_green':  dict(cx=184, N=(184, 8), SL=(65, 55), AL=(52, 160), HL=(30, 915), WL=(52, 360),
                         sleeveL=[(0, 0), (64, 0), (64, 55), (52, 160), (0, 190)], long=True, ease=1.05),
    'wedding':      dict(cx=305, N=(305, 75), SL=(229, 28), AL=(247, 110), HL=(15, 820), WL=(250, 195),
                         sleeveL=[(0, 0), (228, 0), (228, 28), (247, 110), (0, 140)], long=True, ease=1.05, shoulder=230),
    'xiuhe':        dict(cx=327, N=(327, 8), SL=(219, 40), AL=(219, 172), HL=(19, 810), WL=(206, 298),
                         sleeveL=[(0, 0), (218, 0), (219, 173), (0, 260)], long=True, ease=1.2, sleeveSqueeze=0.75),
    'hanfu':        dict(cx=338, N=(338, 8), SL=(258, 28), AL=(258, 166), HL=(75, 820), WL=(275, 195),
                         sleeveL=[(0, 0), (259, 0), (259, 166), (225, 340), (0, 400)], long=True, shoulder=200, ease=1.3, sleeveSqueeze=0.7),
    'mamianqun':    dict(cx=322, N=(322, 8), SL=(219, 45), AL=(234, 205), HL=(109, 810), WL=(236, 280),
                         sleeveL=[(0, 0), (220, 0), (234, 206), (0, 240)], long=True, shoulder=215, ease=1.2, sleeveSqueeze=0.85),
    'yuanlingpao':  dict(cx=335, N=(335, 8), SL=(255, 25), AL=(255, 175), HL=(155, 715), WL=(258, 212),
                         sleeveL=[(0, 0), (256, 0), (256, 178), (0, 178)], long=True, shoulder=190, ease=1.3, sleeveSqueeze=0.75),
    'kimono':       dict(cx=330, N=(330, 10), SL=(210, 45), AL=(210, 205), HL=(213, 925), WL=(218, 260),
                         sleeveL=[(0, 0), (209, 0), (209, 350), (0, 350)], long=True, ease=1.2, sleeveSqueeze=0.75),
    'hanbok':       dict(cx=328, N=(328, 0), SL=(234, 17), AL=(235, 127), HL=(15, 787), WL=(240, 162),
                         sleeveL=[(0, 0), (234, 0), (234, 129), (0, 129)], long=True, shoulder=215, ease=1.3, sleeveSqueeze=0.8),
}


def cut(im, keep=()):
    """
    背景色 = 四边中位色，Lab 色差 > 12 算衣服；裁剪范围按这个蒙版定（标注坐标以它为准）。
    透明度另外去掉衣服投在背景上的淡阴影：比背景略暗（亮度差 2～25）、颜色和背景几乎一样（a、b 差 < 6）
    """
    h, w = im.shape[:2]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.median(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]), axis=0)
    d = np.sqrt(((lab - bg) ** 2).sum(-1))

    dl = bg[0] - lab[..., 0]
    shadow = (dl > 2) & (dl < 25) & (np.abs(lab[..., 1] - bg[1]) + np.abs(lab[..., 2] - bg[2]) < 6)

    def mask(no_shadow):
        m = (d > 12) & ~shadow if no_shadow else d > 12
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        lb, n = ndimage.label(m)
        if n:
            m = lb == (np.argmax(ndimage.sum(m, lb, range(1, n + 1))) + 1)
        holes = ndimage.binary_fill_holes(m) & ~m
        hl, hn = ndimage.label(holes)
        if hn:
            sz = ndimage.sum(holes, hl, range(1, hn + 1))
            m |= np.isin(hl, 1 + np.where(sz < 0.002 * h * w)[0])
        return m
    ref = mask(False)
    ys, xs = np.where(ref)
    pad = 8
    y0, x0 = max(ys.min() - pad, 0), max(xs.min() - pad, 0)
    y1, x1 = min(ys.max() + pad + 1, h), min(xs.max() + pad + 1, w)
    m = mask(True)
    for poly in keep:  # 白衬衫贴着白背景被当成了背景，按领口多边形补回来（坐标是裁剪后的）
        pm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(pm, [np.int32([(x + x0, y + y0) for x, y in poly])], 1)
        m |= pm > 0
    a = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)
    return np.dstack([im, (a * 255).astype(np.uint8)])[y0:y1, x0:x1]


def seam(mask, cx, A, H, right=False):
    """
    腋下以下的袖子边界：只信「衣身外侧隔着一道背景缝还有衣服（袖子）」、且衣身外缘离腋下→下摆连线不超过 45 像素的行
    （袖子上的白条纹、拉链头会让中间那段断开，不能信），取衣身外缘再往外 2 像素；没有缝的行在相邻可信行之间按 y 插值。
    最后一个可信行到衣身下摆（中线上最低的衣服像素）之间整行归衣身，下摆以下（比下摆还低的袖口）归袖子。
    左右两边各算各的（AI 生成的衣服不完全对称，镜像会把下摆算进袖子）
    """
    h, w = mask.shape
    col = np.flatnonzero(mask[:, cx])
    tb = int(col[-1]) if len(col) else h - 1
    known = [(A[1], A[0])]
    for y in range(int(A[1]) + 1, tb + 1):
        row = mask[y]
        if not row[cx]:
            continue
        e = cx
        if right:
            while e < w - 1 and row[e + 1]:
                e += 1
            gap = e < w - 1 and row[e + 1:].any()
        else:
            while e > 0 and row[e - 1]:
                e -= 1
            gap = e > 0 and row[:e].any()
        expect = np.interp(y, [A[1], H[1]], [A[0], H[0]])
        if gap and abs(e - expect) < 45:
            known.append((y, e + 2 if right else e - 2))
    ys = np.array([k[0] for k in known], np.float32)
    xs = np.array([k[1] for k in known], np.float32)
    last = int(ys[-1])
    out = [[round(float(np.interp(y, ys, xs)), 1), y] for y in list(range(int(A[1]) + 4, last, 4)) + [last] if y > A[1]]
    far = w + 50 if right else -50
    out += [[far, last + 1], [far, tb], [cx, tb + 1]]
    return out


def annotation(it, w, h, mask):
    cx = it['cx']
    mir = lambda p: [round(2 * cx - p[0], 1), p[1]]
    ann = {'N': list(it['N'])}
    for k in ('S', 'A', 'H', 'W'):
        if k + 'L' in it:
            ann[k + 'L'] = list(it[k + 'L'])
            ann[k + 'R'] = mir(it[k + 'L'])
    S, A, H = it['SL'], it['AL'], it['HL']
    left = None
    if 'sleeveL' in it:
        left = [list(p) for p in it['sleeveL']]
    elif it.get('sleeve') == 'long':
        # 衣身侧边是弯的：腋下以下沿真实的缝走，直线会把鼓出来的衣身算进袖子，袖子截面就量宽了
        side = seam(mask, cx, A, H)
        ann['sleeveL'] = [[0, 0], [S[0], 0], list(S), list(A)] + side + [[side[-1][0], h + 50], [0, h + 50]]
        SR, AR = ann['SR'], ann['AR']
        side = seam(mask, cx, AR, ann['HR'], right=True)
        ann['sleeveR'] = [[w, 0], [SR[0], 0], SR, AR] + side + [[side[-1][0], h + 50], [w, h + 50]]
    elif it.get('sleeve') == 'short':
        left = [[0, 0], [S[0], 0], list(S), list(A), [0, A[1] + 80]]
    if left:
        ann['sleeveL'] = left
        ann['sleeveR'] = [mir(p) for p in left]
    for k in ('long', 'shoulder', 'ease', 'sleeveSqueeze'):
        if k in it:
            ann[k] = it[k]
    return ann


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for gid, it in ITEMS.items():
        im = cv2.imread(str(SRC / f'{gid}.jpg'), cv2.IMREAD_COLOR)
        rgba = cut(im, it.get('keep', ()))
        cv2.imwrite(str(out / f'{gid}.png'), rgba)
        (out / f'{gid}.json').write_text(json.dumps(annotation(it, rgba.shape[1], rgba.shape[0], rgba[..., 3] >= 128)))
        print(gid, rgba.shape[1], rgba.shape[0])


if __name__ == '__main__':
    main()
