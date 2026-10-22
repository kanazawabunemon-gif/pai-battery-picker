"""Still cards for the submission video (title, problem, how it works, results, summary). 1280x720."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from video.render_demo import FONT_CANDIDATES, font

W, H = 1280, 720
BG = (18, 22, 30)
FG = (240, 240, 240)
ACCENT = (255, 205, 60)
RED = (225, 60, 50)


def _base():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def _center(dr, y, text, size, fill=FG):
    f = font(size)
    dr.text(((W - f.getlength(text)) / 2, y), text, font=f, fill=fill)


def title_card(account: str):
    im, dr = _base()
    _center(dr, 190, "ゴミに紛れた電池を見つけて抜き取る", 58, ACCENT)
    _center(dr, 270, "AIロボットアーム", 58, ACCENT)
    _center(dr, 380, "SO-101 × MuJoCo × YOLO-OBB（向き付き物体検出）", 32)
    _center(dr, 440, "Physical AI 応用講座1 最終課題", 28, (180, 190, 205))
    if account:
        _center(dr, 600, account, 26, (160, 170, 185))
    return im


def problem_card():
    im, dr = _base()
    _center(dr, 80, "なぜ電池なのか", 48, ACCENT)
    lines = [
        "燃えないゴミ・プラごみに混ざったリチウムイオン電池が、",
        "ごみ収集車や処理施設で発火する事故が相次いでいる（環境省も注意喚起）。",
        "",
        "選別の現場では、今も人がゴミの山から手で探して取り除いている。",
        "",
        "→ カメラとAIで電池を見つけ、隠れていればどかしてから、",
        "    ロボットが自動で抜き取る仕組みをシミュレーションで作った。",
    ]
    f = font(31)
    y = 200
    for t in lines:
        dr.text((110, y), t, font=f, fill=RED if t.startswith("→") else FG)
        y += 56
    return im


def pipeline_card():
    im, dr = _base()
    _center(dr, 50, "仕組み", 48, ACCENT)
    boxes = [("真上カメラ", "色画像＋奥行き"), ("AI（YOLO-OBB）", "8種類を向き付きで検出"),
             ("高さ地図", "指を下ろせるか判定"), ("行動を選ぶ", "つかむ／どかす／ずらす／掘る"),
             ("回収", "電池は赤い箱へ")]
    bw, bh, gap = 205, 150, 38
    x = (W - (len(boxes) * bw + (len(boxes) - 1) * gap)) / 2
    y = 230
    f1, f2 = font(23), font(18)
    for i, (a, b) in enumerate(boxes):
        col = RED if i == 4 else (60, 110, 180)
        dr.rounded_rectangle([x, y, x + bw, y + bh], 16, fill=col)
        dr.text((x + (bw - f1.getlength(a)) / 2, y + 30), a, font=f1, fill=(255, 255, 255))
        dr.text((x + (bw - f2.getlength(b)) / 2, y + 88), b, font=f2, fill=(235, 240, 250))
        if i < len(boxes) - 1:
            ax = x + bw + 6
            dr.polygon([(ax, y + bh / 2 - 12), (ax + gap - 12, y + bh / 2), (ax, y + bh / 2 + 12)], fill=ACCENT)
        x += bw + gap
    notes = ["学習データは人手ゼロ：シミュレータの色分け画像から正解ラベルを自動生成（4,500枚）",
             "指を下ろす場所がふさがれていたら、ふさいでいる物を先に一般ごみ箱へ",
             "壁際や密集で挟めない電池は、指先で押さえて空いている方へずらしてからつかむ"]
    f3 = font(25)
    yy = 450
    for t in notes:
        dr.text((120, yy), "・" + t, font=f3, fill=FG)
        yy += 52
    return im


def results_card(vis_bins: dict, e2e: dict):
    """vis_bins: {"0-10%": recall, ...}; e2e: {"AI検出": (removed, total), "完璧な検出": (...)}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for p in FONT_CANDIDATES:
        try:
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            break
        except Exception:
            continue
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), dpi=100, facecolor="#12161e")
    for ax in axes:
        ax.set_facecolor("#12161e")
        for s in ax.spines.values():
            s.set_color("#888")
        ax.tick_params(colors="#ddd", labelsize=13)
    ks = list(vis_bins)
    vs = [100 * (vis_bins[k] or 0) for k in ks]
    axes[0].bar(ks, vs, color="#3c6eb4")
    for i, v in enumerate(vs):
        axes[0].text(i, v + 2, f"{v:.0f}%", ha="center", color="#fff", fontsize=13)
    axes[0].set_ylim(0, 112)
    axes[0].set_title("隠れ具合ごとの電池の発見率", color="#ffcd3c", fontsize=17)
    axes[0].set_xlabel("見えている割合", color="#ddd", fontsize=13)
    names = list(e2e)
    rates = [100 * a / b for a, b in e2e.values()]
    axes[1].bar(names, rates, color=["#e13c32", "#888"][:len(names)])
    for i, (r, (a, b)) in enumerate(zip(rates, e2e.values())):
        axes[1].text(i, r + 2, f"{r:.0f}%  ({a}/{b})", ha="center", color="#fff", fontsize=13)
    axes[1].set_ylim(0, 112)
    axes[1].set_title("48トレイでの危険物の回収率", color="#ffcd3c", fontsize=17)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    chart = Image.open(buf).convert("RGB")
    im, dr = _base()
    _center(dr, 30, "結果（シミュレーション）", 44, ACCENT)
    im.paste(chart.resize((1200, 525)), (40, 110))
    _center(dr, 655, "見間違いゼロの場合＝AIの代わりにシミュレータの正解を使った比較用。差は3ポイント → 取り残しの大部分は掴む側", 21, (180, 190, 205))
    return im


def summary_card():
    im, dr = _base()
    _center(dr, 90, "まとめ", 48, ACCENT)
    lines = ["・合成画像だけで学習したAIが、半分隠れた電池も約9割見つけた",
             "・見つけるだけでなく「どかす」「ずらす」「掘る」を選んで回収まで自動化",
             "・腕は数万円で組めるオープンハードウェア SO-101。実機へ移せる構成",
             "",
             "今後：実写画像での検証（sim2real）、X線など他センサとの併用、",
             "      柔らかいゴミ（袋・紙）の扱い、より速い動作"]
    f = font(31)
    y = 200
    for t in lines:
        dr.text((120, y), t, font=f, fill=FG)
        y += 60
    return im


def to_frames(img: Image.Image, seconds: float, fps: int = 20):
    a = np.asarray(img)
    return [a] * int(seconds * fps)


def data_card(train_img: Image.Image, pred_img: Image.Image, n_images=4500):
    """Left: an auto-labelled training image. Right: the trained detector on an unseen image."""
    im, dr = _base()
    _center(dr, 36, "AIの学習：正解ラベルは人手ゼロで自動生成", 44, ACCENT)
    w, h = 600, 450
    for i, (img, cap) in enumerate([(train_img, "合成画像4,500枚（学習4,000・検証500）。枠は自動で付けた正解"),
                                    (pred_img, "学習したAI（YOLO-OBB）が初めて見る画像で検出")]):
        x = 26 + i * (w + 28)
        im.paste(img.convert("RGB").resize((w, h)), (x, 120))
        f = font(20)
        dr.text((x + (w - f.getlength(cap)) / 2, 585), cap, font=f, fill=FG)
    _center(dr, 640, "照明・カメラの取り付け誤差・ゴミの種類と積み方を毎回ランダムに変えて、実物との差に備える", 22,
            (180, 190, 205))
    return im
