import io
import base64
from PIL import Image
import matplotlib.pyplot as plt

# Utility functions for image encoding/decoding

def pil_to_base64(img: Image.Image) -> str:
    """
    Convert a PIL Image to a base64-encoded PNG string.
    """
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib Figure to a base64-encoded PNG string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def base64_to_bytes(b64str: str) -> bytes:
    """
    Decode a base64-encoded string to raw bytes.
    """
    return base64.b64decode(b64str)


def bytes_to_pil(img_bytes: bytes) -> Image.Image:
    """
    Convert raw image bytes to a PIL Image.
    """
    return Image.open(io.BytesIO(img_bytes))
