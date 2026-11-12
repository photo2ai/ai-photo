# 把 ai.py、museum.py 的输出打成 App 按需下载的衣服包，并生成随 APK 带的小缩略图。
#   衣服包 outfits_v1.zip：每件 <id>.webp（带透明，长边不超过 1024）+ <id>.json（关键点，坐标随图缩放）
#   缩略图 app/src/main/res/drawable-nodpi/outfit_<id>.webp（160×160）
# 用法: python pack.py <ai.py 输出目录> <museum.py 输出目录> <zip 输出路径> <res 目录>
# zip 里的时间戳固定，同样的输入每次打出来的文件（SHA-256）一样；列表要和 core/Outfits.kt 的 ITEMS 一致
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

# (App 里的 id, 来源：ai / museum, 来源里的文件名)
IDS = [(n, 'ai', n) for n in [
    'tee_white', 'tee_black', 'tee_stripe', 'polo', 'shirt_white', 'shirt_plaid', 'hoodie', 'sweater', 'sport',
    'cardigan', 'denim_jacket', 'bomber', 'down_jacket', 'trench',
    'suit', 'jk',
    'dress_black', 'gown', 'qipao_green',
    'wedding', 'xiuhe', 'groom',
    'hanfu', 'mamianqun', 'yuanlingpao', 'tangzhuang',
    'kimono', 'hanbok',
]] + [('qipao', 'museum', 'qipao'), ('wedding_satin', 'museum', 'wedding'), ('dragon', 'museum', 'dragon'), ('qing', 'museum', 'qing')]
MAX_SIDE = 1024
THUMB = 160


def scale_ann(ann, k):
    out = {}
    for key, v in ann.items():
        if key in ('shoulder',):
            out[key] = round(v * k, 1)
        elif isinstance(v, list) and v and isinstance(v[0], list):
            out[key] = [[round(x * k, 1), round(y * k, 1)] for x, y in v]
        elif isinstance(v, list):
            out[key] = [round(v[0] * k, 1), round(v[1] * k, 1)]
        else:
            out[key] = v
    return out


def main():
    dirs = {'ai': Path(sys.argv[1]), 'museum': Path(sys.argv[2])}
    zip_path, res = Path(sys.argv[3]), Path(sys.argv[4])
    thumbs = res / 'drawable-nodpi'
    thumbs.mkdir(parents=True, exist_ok=True)
    for old in thumbs.glob('outfit_*.webp'):
        old.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
        for gid, src, name in IDS:
            im = Image.open(dirs[src] / f'{name}.png').convert('RGBA')
            ann = json.loads((dirs[src] / f'{name}.json').read_text())
            k = min(1.0, MAX_SIDE / max(im.size))
            if k < 1:
                im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
                ann = scale_ann(ann, k)
            buf = io.BytesIO()
            im.save(buf, 'WEBP', quality=88, method=6, alpha_quality=100)
            for fn, data in ((f'{gid}.webp', buf.getvalue()), (f'{gid}.json', json.dumps(ann, ensure_ascii=False).encode())):
                z.writestr(zipfile.ZipInfo(fn, date_time=(2026, 1, 1, 0, 0, 0)), data)
            # 缩略图：裁到衣服范围，等比放进正方形
            t = im.crop(im.getbbox())
            t.thumbnail((THUMB, THUMB), Image.LANCZOS)
            sq = Image.new('RGBA', (THUMB, THUMB), (0, 0, 0, 0))
            sq.paste(t, ((THUMB - t.width) // 2, (THUMB - t.height) // 2))
            sq.save(thumbs / f'outfit_{gid}.webp', 'WEBP', quality=85, method=6)
            print(gid, im.size, len(buf.getvalue()) // 1024, 'KB')
    data = zip_path.read_bytes()
    print(zip_path.name, len(data), hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
