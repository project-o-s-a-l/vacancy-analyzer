"""Bounded, in-memory text extraction; uploaded files are never executed or saved."""

import re
from io import BytesIO
from pathlib import PurePath
from zipfile import BadZipFile, ZipFile

from lxml import etree

from domain.resume import MAX_RESUME_TEXT

MAX_RESUME_BYTES = 2 * 1024 * 1024
MAX_PAGES = 20


class ResumeError(ValueError):
    pass


class ResumeBuffer(BytesIO):
    def write(self, data: bytes) -> int:
        if self.tell() + len(data) > MAX_RESUME_BYTES:
            raise ResumeError("Резюме должно быть не больше 2 МБ.")
        return super().write(data)


def normalize_resume_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in text:
        raise ResumeError("Не удалось прочитать текст. Отправьте PDF, DOCX или TXT.")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\u200b\ufeff]", "", text)
    text = "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()
    if len(text) > MAX_RESUME_TEXT:
        raise ResumeError("В резюме слишком много текста: максимум 60 000 символов.")
    if sum(char.isalpha() for char in text) < 20:
        raise ResumeError(
            "В файле недостаточно читаемого текста. Для скана нужен PDF с текстовым слоем "
            "либо резюме в DOCX или TXT."
        )
    return text


def resume_format(filename: str | None) -> str:
    extension = PurePath(filename or "").suffix.lower().lstrip(".")
    if extension not in {"pdf", "docx", "txt"}:
        raise ResumeError("Отправьте резюме файлом PDF, DOCX или TXT размером до 2 МБ.")
    return extension


def read_resume(data: bytes, filename: str | None) -> tuple[str, str]:
    kind = resume_format(filename)
    if not data or len(data) > MAX_RESUME_BYTES:
        raise ResumeError("Резюме должно быть непустым файлом размером до 2 МБ.")
    try:
        if kind == "txt":
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("cp1251")
        elif kind == "docx":
            text = _docx_text(data)
        else:
            text = _pdf_text(data)
    except ResumeError:
        raise
    except Exception as exc:
        # Parser diagnostics may contain uploaded text. Return only a fixed message.
        raise ResumeError("Не удалось прочитать файл. Проверьте его или отправьте TXT.") from exc
    return normalize_resume_text(text), kind


def _docx_text(data: bytes) -> str:
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 500 or sum(item.file_size for item in entries) > 10 * 1024 * 1024:
                raise ResumeError("DOCX слишком большой после распаковки. Отправьте TXT или PDF.")
            xml = archive.read("word/document.xml")
    except (BadZipFile, KeyError) as exc:
        raise ResumeError(
            "Файл не является читаемым DOCX. Старый формат DOC не поддерживается."
        ) from exc
    # Reject DTDs rather than expanding document-defined entities.
    if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
        raise ResumeError("Не удалось прочитать структуру DOCX. Отправьте TXT или PDF.")
    root = etree.fromstring(xml, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join(
        "".join(
            node.text or "" if node.tag == ns + "t" else "\n" if node.tag == ns + "br" else " "
            for node in paragraph.iter()
            if node.tag in {ns + "t", ns + "br", ns + "tab"}
        )
        for paragraph in root.iter(ns + "p")
    )


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    if not data.startswith(b"%PDF-"):
        raise ResumeError("Файл не является читаемым PDF.")
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        raise ResumeError("PDF защищён паролем. Отправьте копию без пароля или TXT.")
    if len(reader.pages) > MAX_PAGES:
        raise ResumeError("Резюме должно содержать не больше 20 страниц.")
    parts = []
    total = 0
    for page in reader.pages:
        part = page.extract_text() or ""
        total += len(part)
        if total > MAX_RESUME_TEXT:
            raise ResumeError("В резюме слишком много текста: максимум 60 000 символов.")
        parts.append(part)
    return "\n".join(parts)
