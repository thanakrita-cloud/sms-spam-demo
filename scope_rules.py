"""กติกาตรวจว่าข้อความอยู่ในขอบเขตของเดโมหรือไม่ (ไม่มี dependency ภายนอก)"""

MAX_CHARS = 1000
HARD_LIMIT = 5000


def has_thai(text):
    return any('\u0e00' <= char <= '\u0e7f' for char in text)


def mostly_non_latin(text):
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for char in letters if not ('a' <= char.lower() <= 'z'))
    return non_latin / len(letters) > 0.5


def scope_issue(text, known_features=None):
    """คืนเหตุผลที่ประเมินไม่ได้ หรือ None เมื่อข้อความผ่านกติกาขอบเขต."""
    if has_thai(text) or mostly_non_latin(text):
        return 'รองรับเฉพาะ SMS ภาษาอังกฤษ'
    if len(text) > HARD_LIMIT:
        return f'ข้อความยาวเกิน {HARD_LIMIT:,} ตัวอักษร และไม่ใช่ SMS สั้นตามขอบเขตของระบบ'
    if known_features == 0:
        return 'ไม่พบคำหรือคู่คำที่โมเดลรู้จัก จึงมีเพียงค่าเริ่มต้นของโมเดล'
    return None
