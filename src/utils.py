CLC_MAJOR_CLASSES = {
    "A": "马列主义、毛泽东思想",
    "B": "哲学、宗教",
    "C": "社会科学总论",
    "D": "政治、法律",
    "E": "军事",
    "F": "经济",
    "G": "文化、科学、教育、体育",
    "H": "语言、文字",
    "I": "文学",
    "J": "艺术",
    "K": "历史、地理",
    "N": "自然科学总论",
    "O": "数理科学和化学",
    "P": "天文学、地球科学",
    "Q": "生物科学",
    "R": "医药、卫生",
    "S": "农业科学",
    "T": "工业技术",
    "U": "交通运输",
    "V": "航空、航天",
    "X": "环境科学",
    "Z": "综合性图书",
}

NUM_CLASSES = len(CLC_MAJOR_CLASSES)
LABEL_TO_CLASS = sorted(CLC_MAJOR_CLASSES.keys())
CLASS_TO_LABEL = {c: i for i, c in enumerate(LABEL_TO_CLASS)}


def extract_major_class(clc_code: str) -> str | None:
    if not isinstance(clc_code, str) or not clc_code.strip():
        return None
    first = clc_code.strip()[0].upper()
    return first if first in CLC_MAJOR_CLASSES else None


def class_name(code: str) -> str:
    return CLC_MAJOR_CLASSES.get(code, "未知类别")


def class_to_label(code: str) -> int:
    return CLASS_TO_LABEL[code]


def label_to_class(label: int) -> str:
    return LABEL_TO_CLASS[label]
