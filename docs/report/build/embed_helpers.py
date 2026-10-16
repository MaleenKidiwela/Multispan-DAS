import base64, io, re
from PIL import Image
F = "/data/wsd02/multispan_dvv/outputs_100m/figs/"
def uri(name, w=1500, q=82):
    im = Image.open(F + name).convert("RGB")
    if im.width > w:
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def fig(name, alt, cap, w=1500):
    return f'  <figure><img src="{uri(name, w)}" alt="{alt}"><figcaption>{cap}</figcaption></figure>\n'

