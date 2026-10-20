"""Screening of an uploaded file, for a future upload route (Studio S8, Studio addendum rule B).

v0.1 has no upload route. This module is the check such a route must call on every file BEFORE the file is stored,
sent anywhere or read by anything else:

    verdict = screen_upload(filename, content_type, data, text)
    if not verdict.allowed:
        answer 422 with verdict.message (it never repeats the file's text) and drop the bytes

`text` is the file's text as the route extracted it. For a PDF that is the text layer of every page PLUS the OCR text
of every page that has no text layer: a scanned page is an image, so its text layer is empty (docintel/pdftext.py
treats a page with fewer than MIN_TEXT_CHARS = 20 characters as having no text layer and sends it to OCR). For an
image it is the OCR text. For CSV and plain-text files the module decodes `data` itself, and also screens `text` when
it is given. text=None for a PDF or an image means "not read", and such a file is refused (fail closed): a file whose
text was never read cannot be shown to be free of SSNs, tax returns and account numbers. A PDF whose text holds fewer
than MIN_PDF_TEXT_CHARS (20) visible characters is refused the same way: that is what a scanned PDF (a paper Form
1040 or bank statement put through a scanner) gives when nobody ran OCR on it, and the screen cannot tell that from
a PDF that was OCR'd and holds almost no text. An image that holds no text (a site photo) is passed with text="":
an image has no text layer, so a text for it can only come from OCR; the route must pass text=None for an image it
did not OCR.

What is refused (refusal_codes, always in this order). The first four are gates: a file stopped at a gate is not
read further and gets exactly one code.

1. too_large: more than MAX_UPLOAD_BYTES bytes (10 MiB, a design choice), or a file name longer than
   MAX_FILENAME_CHARS characters (255, the longest name common file systems allow; a design choice). The route
   should also stop reading the request body at that cap (web/security.py limits request bodies per path prefix);
   this check is the second line. Checked again once the text is known (after gate 4): a text longer than
   MAX_TEXT_CHARS characters (1,000,000, a design choice), whether given by the route or decoded from a CSV or
   plain-text file, is refused as too_large too. Every character is read (the normalization in security.normalize
   goes character by character in Python), so the text is capped to keep one screen short; screening only part of a
   longer text would let the rest through unchecked. The cap keeps a screen short only because every check takes
   time in proportion to the length of the text: the patterns are written so that the regex engine never retries a
   run of whitespace in more than one way (possessive "\\s*+", see UPLOAD_LABELED_ACCOUNT), and the agency line of
   rule 8 is matched to form numbers in one pass. A test feeds long adversarial texts (a label followed by a long run
   of spaces, many agency lines and form numbers) at two sizes and checks that eight times the text takes far less
   than 64 times as long.
2. empty: zero bytes.
3. file_type: the file is not one of ALLOWED_KINDS, or its name, its declared type and its content disagree. A kind
   is accepted only when all three agree:
   - PDF (.pdf, application/pdf): the data starts with "%PDF-", the file header of ISO 32000-1:2008, 7.5.2. A
     header further into the file, which some readers tolerate, is refused;
   - PNG (.png, image/png): the 8-byte PNG signature 89 50 4E 47 0D 0A 1A 0A (W3C PNG Specification, 5.2);
   - JPEG (.jpg or .jpeg, image/jpeg): FF D8 FF, the start-of-image marker FF D8 followed by the next marker
     (ITU-T T.81, Annex B);
   - CSV (.csv; declared as text/csv, as text/plain, or as application/vnd.ms-excel, the type Windows browsers often
     send for a .csv file) and plain text (.txt, text/plain): text has no magic bytes, so the data must decode as
     UTF-8 (a byte order mark is allowed), hold no NUL byte and not start with the PDF, PNG or JPEG signature. A CSV
     saved in another encoding (for example Windows-1252) is refused: save it as "CSV UTF-8".
   The declared type is compared without its parameters ("text/csv; charset=utf-8" is text/csv) and ignoring case,
   and so is the extension (only the last one counts: "statement.pdf.exe" is an .exe). An empty or generic declared
   type (application/octet-stream) is refused.
4. unreadable: a PDF, PNG or JPEG given with text=None, or a PDF whose whole text holds fewer than
   MIN_PDF_TEXT_CHARS visible characters (whitespace and invisible characters are not counted; see above). 20 is the
   number docintel/pdftext.py uses for one page: a page with fewer printed characters has no text layer. The
   short-PDF test runs after the text cap of rule 1, so it never counts more than MAX_TEXT_CHARS characters. A PDF
   that is only a photo is refused too: upload the photo as a PNG or JPEG.

Then the content: the file name and the file's text. Each is read as written and after a light normalization
(invisible characters removed, NFKC, lookalike letters and typographic dashes folded; uwcopilot.security.normalize),
so an SSN written in fullwidth digits, with en dashes or with a zero-width space inside is still found. Runs of
whitespace are NOT collapsed (security.normalize.normalize_text would collapse them), so numbers in separate columns
of a table ("120    45    2026") are not joined into an SSN shape. The file name is also read with its separators
turned into spaces (agent.injection.code_as_words: "Form_1040_2025.pdf" is read as "Form 1040 2025 pdf").

5. ssn: an SSN-shaped number. First the shapes typed text refuses (web/services/text_checks.py): 123-45-6789
   (observability.redaction.SSN) and 123 45 6789 or mixed separators (text_checks.SPACED_SSN). Uploads also refuse
   dashes with up to two spaces or tabs on each side ("123 - 45 - 6789") and these shapes right after a word and a
   hyphen ("SSN-123-45-6789") (SEPARATED_SSN), and the digits one per box with the dashes kept, as the comb fields
   of a paper form extract ("1 2 3 - 4 5 - 6 7 8 9", COMB_SSN). Then nine digits, also one per box
   ("1 2 3 4 5 6 7 8 9"), after an SSN label: "SSN", "SS#", "Social Security" (also "Social-Security"), "Social
   Security Number" or "Social Security No.", then up to 12 characters of short filler (letters, spaces and
   : # = ( ) . - /, as in "SSN/TIN: ", "Social Security Number (SSN): ", "SSN No. " or "SSN - ") and any run of
   spaces, ":" or "#" (LABELED_SSN). Only the shape is checked, not whether the Social Security Administration could
   issue the number, so a part number written 123-45-6789 is refused too.
6. bank_account: a labeled account number (UPLOAD_LABELED_ACCOUNT): the word "account", "acct", "a/c" or "acc"
   (only before "." or "#", as in "Acc# "), an optional ".", optionally "no.", "number", "#" or "ID", then
   optionally ":", "#", "=", "-" (an en or em dash is folded to "-") or "is", then a number of at least five digits,
   spaces or dashes that starts and ends with a digit ("Account Number: 000123456789", "acct 1234-5678",
   "Acct. No. 12345678", "Account number is 12345678"). This widens the typed-text rule
   (observability.redaction.LABELED_ACCOUNT) and runs in linear time; typed text needs it less, because it also
   refuses every run of 8 to 17 digits, which uploads do not (see "Not refused"). A masked number ("Account ending
   in 1234", "Account ****5678") is not refused.
7. routing_number: a nine-digit number that has the form of an ABA routing transit number: its first two digits are
   in ROUTING_PREFIXES (01-12, 21-32, 61-72 or 80) AND it passes the ABA check digit,
   3 x (d1 + d4 + d7) + 7 x (d2 + d5 + d8) + (d3 + d6 + d9) is divisible by 10. The two tests cut false positives:
   37 of the 100 two-digit prefixes are allowed and, for any first eight digits, exactly one ninth digit of ten
   passes the checksum, so a random nine-digit number passes both with probability 37/100 x 1/10 (a test checks
   both counts). Digits that are part of a longer digit run, of a dashed number (a ZIP+4 such as 60601-1234), of a
   decimal number, or that follow a "$" (an amount) are not candidates. A comma next to the digits does not stop
   them: in a CSV it is a delimiter ("SYNTHETIC Bank,123456780,4500"), and a nine-digit run can never be one group of
   a comma-grouped number such as 123,456,780. The prefix 00 (reserved for the U.S. government) is left out:
   zero-padded ids such as 000000000 are common in tables, and a government routing number is not a person's bank
   detail.
   Source of the checksum and the prefix ranges: the Wikipedia article "ABA routing transit number" (read
   2026-10-20), which cites Bankers' Hotline (2004), "Training Page: Learning the Bank Numbering System", for the
   check digit and gives no citation for the prefix list. Neither was checked against the routing number registrar's
   own policy text (LexisNexis Risk Solutions, registrar for the American Bankers Association).
8. tax_return: an IRS form marker, matched ignoring case. Any run of whitespace (spaces, a line break) may stand
   between the words, and the space after "Form" or "Schedule" may be missing ("form1040"):
   - "Form 1040" (also 1040-SR, 1040-NR, 1040-X, 1040-ES), "Form 1120" (also 1120-S), "Form 1065";
   - "1120-S" on its own (so an address such as "Suite 1120-S" is refused too);
   - the title "U.S. Individual Income Tax Return" (the dots optional);
   - "Schedule K-1" (also "Schedule K1");
   - "Schedule C" followed by "(Form 1040)" or by its title "Profit or Loss From Business". "Schedule C" alone is
     not a marker: leases and loan agreements name their attachments Schedule A, B, C;
   - "Department of the Treasury" then "Internal Revenue Service" (any dash, comma or line break between them)
     within NEAR_CHARS characters of a form number ("Form" and three or four digits, such as "Form 941").
   A text that only mentions a form ("we can send our Form 1040 later") is refused too: the screen cannot tell a
   mention from the form itself.

Not refused (each has a test): phone numbers ((312) 555-0147, 312-555-0147, 3125550147), ZIP+4 codes (60601-1234,
606011234), 11-digit census tract ids, dollar amounts ($1,250,000.00, $250000), dates (2026-10-20, 10/20/2026,
20261020), an EIN (12-3456789, a business id that typed text allows too). Typed text also refuses every run of 8 to 17
digits (redaction.LONG_DIGITS); uploads do not, because CSV and PDF tables are full of such runs (tract ids, FIPS
codes, dates and phone numbers written without separators, invoice numbers). The price is the first known limit
below.

Injection is a FLAG, never a refusal: uploaded text is data, never instructions. The text and the file name are read
by the agent's tripwire (uwcopilot.agent.injection.detect_untrusted: as written and after
security.normalize.detection_text, which also decodes a message hidden in Unicode tag characters). A hit sets
injection_flagged and lists the pattern names; the file is still allowed. The route must store the flag with the
file and keep a flagged file away from any language model (the rule text_checks.py states for the idea text).

The verdict holds no text from the file, not even an excerpt: only codes, pattern names and a fixed message, so it
can be logged and returned as it is.

Known limits (a future route must not rely on this screen for more):
- an unlabeled account number, or an SSN written as nine bare digits without a label, is not found; nor is a column
  of them under a header (an "ssn" header on the first row of a CSV), because the label must come right before the
  number;
- these SSN writings are not found either: dotted (123.45.6789), and the digits one per box without dashes and
  without a label ("1 2 3 4 5 6 7 8 9"), because a row of numbered table columns looks the same;
- an account number written with dots (0001.2345.6789) is not found: allowing dots in the number would also refuse an
  amount such as "account 1250.00";
- the magic-byte check proves only the first bytes; it is not a parser, so a malformed or polyglot file can pass.
  Open uploads only with a maintained parser, never execute them, and serve them back only as downloads;
- the screen is only as good as the text it is given: an OCR error ("Form 1O40" with a letter O) defeats it;
- the PDF minimum (gate 4) applies to the whole text, not page by page: a PDF with one typed page and one scanned
  page passes on the typed page alone unless the route OCRs the scanned page, and so does a scan with a few words of
  text layer on top. Only the route can see the pages, so it must OCR every page without a text layer;
- the markers are in English and for US forms only.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from uwcopilot.agent.injection import ALL_PATTERNS, code_as_words, detect_untrusted
from uwcopilot.observability.redaction import SSN
from uwcopilot.security.normalize import fold_lookalikes, strip_invisible
from uwcopilot.web.services.text_checks import PRIVACY_NOTICE, SPACED_SSN

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB, a design choice
MAX_TEXT_CHARS = 1_000_000  # characters of text screened per file, a design choice (module docstring, rule 1)
MAX_FILENAME_CHARS = 255  # the longest file name common file systems allow (ext4, NTFS), a design choice
MIN_PDF_TEXT_CHARS = 20  # docintel.pdftext.MIN_TEXT_CHARS, applied to a PDF's whole text (gate 4), a design choice
NEAR_CHARS = 300  # how close "Department of the Treasury - Internal Revenue Service" must be to a form number

RefusalCode = Literal[
    "too_large", "empty", "file_type", "unreadable", "ssn", "bank_account", "routing_number", "tax_return"
]
FileKindName = Literal["pdf", "png", "jpeg", "csv", "text"]

PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
BINARY_MAGICS: tuple[bytes, ...] = (PDF_MAGIC, PNG_MAGIC, JPEG_MAGIC)


@dataclass(frozen=True)
class FileKind:
    """One accepted file type: its extensions, the declared types accepted for it and its leading bytes."""

    name: FileKindName
    extensions: tuple[str, ...]
    content_types: tuple[str, ...]
    magic: bytes | None  # None for a text format: checked by decoding instead


ALLOWED_KINDS: tuple[FileKind, ...] = (
    FileKind("pdf", (".pdf",), ("application/pdf",), PDF_MAGIC),
    FileKind("png", (".png",), ("image/png",), PNG_MAGIC),
    FileKind("jpeg", (".jpg", ".jpeg"), ("image/jpeg",), JPEG_MAGIC),
    FileKind("csv", (".csv",), ("text/csv", "text/plain", "application/vnd.ms-excel"), None),
    FileKind("text", (".txt",), ("text/plain",), None),
)

# Rule 5 (the typed-text shapes SSN and SPACED_SSN are checked too). One separator inside an SSN: a dash with up to
# two spaces or tabs on each side ("123 - 45 - 6789"), or one space ("123 45 6789").
SSN_SEPARATOR = r"(?:[ \t]{0,2}-[ \t]{0,2}| )"
# An SSN with those separators. It may follow a word and a hyphen ("SSN-123-45-6789"; the typed-text shapes refuse
# any hyphen before the digits). It is not found inside a longer number: a digit, or a digit and a dash, right before
# it, or a digit, or a dash and a digit, right after it.
SEPARATED_SSN = re.compile(rf"(?<!\d)(?<!\d-)\d{{3}}{SSN_SEPARATOR}\d{{2}}{SSN_SEPARATOR}\d{{4}}(?!-?\d)")
# The same with each digit in its own box and the dashes kept, as the comb fields of a paper form extract
# ("1 2 3 - 4 5 - 6 7 8 9").
COMB_SSN = re.compile(r"(?<!\d)\d \d \d[ \t]{0,2}-[ \t]{0,2}\d \d[ \t]{0,2}-[ \t]{0,2}\d \d \d \d(?!\d)")
# Nine digits (or nine digits one per box: "1 2 3 4 5 6 7 8 9") after an SSN label. Between the label and the digits
# stand at most 12 characters of short filler (letters, spaces and : # = ( ) . - /, as in "SSN/TIN: ",
# " (SSN): " or " No. "), then any run of spaces, ":" or "#". The filler is bounded and the run is possessive
# ("*+", see UPLOAD_LABELED_ACCOUNT), so the time stays linear. "ssn" must not run on into a word ("ssnake").
LABELED_SSN = re.compile(
    r"\b(?:ssn(?![a-z])|ss\s*#|social[\s-]+security(?:\s+(?:number|no\.?))?)"
    r"[\s:#=().\-/a-z]{0,12}?[\s:#]*+\d(?: ?\d){8}(?!\d)",
    re.IGNORECASE,
)
# Rule 6: the typed-text rule (observability.redaction.LABELED_ACCOUNT), widened for uploads and written so that it
# runs in linear time. In "\s*+" the "+" makes the run of whitespace possessive: once read it is never given back. The
# typed-text pattern has two plain "\s*" next to each other (around an optional ":" or "#"), and on "account" followed
# by a long run of spaces the regex engine tries every way to split that run between them, which takes time that
# grows with the square of the run (UP-1). Nothing after a whitespace run can be whitespace, so giving it back never
# finds a match that keeping it misses. Widened (UP-7): a "." after the label ("Acct. No."), "acc" before "." or "#"
# ("Acc# "; plain "ACC 2025-26" is a sports conference), "ID" as well as "no.", "number" and "#", and "=", "-" or
# "is" before the number ("Account Number - 1234...", "account number is 1234...").
UPLOAD_LABELED_ACCOUNT = re.compile(
    r"\b(?:account|acct|acc(?=[.#])|a/c)\.?(?:\s*+(?:no\.?|number|#|id))?\s*+(?:(?:[:#=-]|\bis\b)\s*+)?"
    r"\d[\d -]{3,}\d",
    re.IGNORECASE,
)
# Exactly nine digits, not inside a longer or dashed number, a decimal number, or an amount (rule 7). A comma next to
# the digits is a CSV delimiter, not digit grouping: a nine-digit run can never be one group of a grouped number.
NINE_DIGITS = re.compile(r"(?<![\d$-])(?<!\d\.)\d{9}(?![\d-]|\.\d)")
ROUTING_PREFIXES: frozenset[int] = frozenset([*range(1, 13), *range(21, 33), *range(61, 73), 80])
ABA_WEIGHTS: tuple[int, ...] = (3, 7, 1, 3, 7, 1, 3, 7, 1)

# Rule 8: markers that are enough on their own. The K-1 marker uses possessive "\s*+" for the reason given at
# UPLOAD_LABELED_ACCOUNT ("schedule\s*k\s*-?\s*1" had two "\s*" next to each other when the dash is missing).
TAX_RETURN_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bform\s*1040(?!\d)", re.IGNORECASE),
    re.compile(r"\bform\s*1120(?!\d)", re.IGNORECASE),
    re.compile(r"\bform\s*1065(?!\d)", re.IGNORECASE),
    re.compile(r"(?<![\w-])1120\s*-\s*s\b", re.IGNORECASE),
    re.compile(r"\bu\.?\s*s\.?\s*individual\s+income\s+tax\s+return\b", re.IGNORECASE),
    re.compile(r"\bschedule\s*+k\s*+(?:-\s*+)?1(?!\d)", re.IGNORECASE),
    re.compile(r"\bschedule\s*c\b[\s(]*(?:form\s*1040|profit\s+or\s+loss\s+from\s+business)", re.IGNORECASE),
)
# Rule 8: the agency line counts only near a form number.
TREASURY_IRS = re.compile(r"\bdepartment\s+of\s+the\s+treasury[\s,-]*internal\s+revenue\s+service\b", re.IGNORECASE)
FORM_NUMBER = re.compile(r"\bform\s*\d{3,4}(?!\d)", re.IGNORECASE)

GATE_MESSAGES: dict[RefusalCode, str] = {
    "too_large": (
        f"The file is larger than the upload limit ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB, or {MAX_TEXT_CHARS:,} "
        f"characters of text), or its name is longer than {MAX_FILENAME_CHARS} characters."
    ),
    "empty": "The file is empty.",
    "file_type": (
        "Only PDF, PNG, JPEG, CSV and plain-text files are accepted (CSV and text in UTF-8), and the file's name, "
        "declared type and content must match."
    ),
    "unreadable": (
        "The file's text could not be read, so it could not be checked for SSNs, tax returns or bank account "
        "numbers. It was not accepted."
    ),
}
FOUND_MESSAGES: dict[RefusalCode, str] = {
    "ssn": "an SSN-shaped number",
    "bank_account": "a labeled account number",
    "routing_number": "a bank routing number",
    "tax_return": "what looks like a tax return or an IRS tax form",
}
ACCEPTED_MESSAGE = "The file passed the upload checks."
FLAGGED_MESSAGE = (
    "Some of its text reads like instructions to a reviewer or an AI; it is marked as flagged and treated as data "
    "only, never as instructions."
)


class UploadVerdict(BaseModel):
    """What screen_upload decided about one file. It holds no text from the file, so it is safe to log and return."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    refusal_codes: tuple[RefusalCode, ...] = ()
    file_kind: FileKindName | None = None
    injection_flagged: bool = False
    injection_patterns: tuple[str, ...] = ()
    message: str

    @model_validator(mode="after")
    def allowed_means_no_refusal(self) -> "UploadVerdict":
        """A file is allowed exactly when nothing refused it."""
        if self.allowed == bool(self.refusal_codes):
            raise ValueError("allowed must be true exactly when there are no refusal codes")
        return self


# --- file type (gate 3) ---


def file_extension(filename: str) -> str:
    """The last extension of the file name, lowercased ("Plan.PDF" -> ".pdf"; no extension -> "")."""
    return PurePosixPath(filename.strip()).suffix.lower()


def declared_type(content_type: str) -> str:
    """The declared media type without parameters, lowercased ("Text/CSV; charset=utf-8" -> "text/csv")."""
    return content_type.split(";", 1)[0].strip().lower()


def decode_text(data: bytes) -> str | None:
    """data as UTF-8 text (a byte order mark is dropped), or None when it is not text: it starts with a PDF, PNG or
    JPEG signature, is not valid UTF-8, or holds a NUL byte."""
    if data.startswith(BINARY_MAGICS):
        return None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    if "\x00" in text:
        return None
    return text


def content_matches(kind: FileKind, data: bytes) -> bool:
    """True when the bytes are what the kind says: its magic bytes first, or decodable text for CSV and text."""
    if kind.magic is not None:
        return data.startswith(kind.magic)
    return decode_text(data) is not None


def kind_of(filename: str, content_type: str, data: bytes) -> FileKind | None:
    """The allowed kind on which the extension, the declared type and the content all agree (None when none does)."""
    extension = file_extension(filename)
    declared = declared_type(content_type)
    for kind in ALLOWED_KINDS:
        if extension in kind.extensions and declared in kind.content_types and content_matches(kind, data):
            return kind
    return None


# --- content checks (rules 5 to 8) ---


def readings(text: str) -> list[str]:
    """text as written and its lightly normalized form, without duplicates (module docstring, "Then the content").

    Invisible characters are removed, NFKC is applied (fullwidth digits become ASCII), invisible characters are
    removed again, then lookalike letters and typographic dashes are folded. Whitespace is kept as it is. ASCII text
    cannot change under any of these steps, so it is returned at once.
    """
    if text.isascii():
        return [text]
    compatible = strip_invisible(unicodedata.normalize("NFKC", strip_invisible(text)))
    folded = fold_lookalikes(compatible)
    if folded == text:
        return [text]
    return [text, folded]


def has_ssn(text: str) -> bool:
    """Rule 5: an SSN-shaped number (dashes, spaces, spaced dashes, comb boxes) or nine digits after an SSN label."""
    for pattern in (SSN, SPACED_SSN, SEPARATED_SSN, COMB_SSN, LABELED_SSN):
        if pattern.search(text):
            return True
    return False


def has_bank_account(text: str) -> bool:
    """Rule 6: a labeled account number (UPLOAD_LABELED_ACCOUNT: the typed-text rule, widened, in linear time)."""
    return UPLOAD_LABELED_ACCOUNT.search(text) is not None


def aba_checksum_ok(digits: str) -> bool:
    """The ABA check digit test on nine digits: weights 3, 7, 1 repeated, the weighted sum divisible by 10."""
    if len(digits) != len(ABA_WEIGHTS):
        return False
    total = 0
    for weight, digit in zip(ABA_WEIGHTS, digits, strict=True):
        total += weight * int(digit)
    return total % 10 == 0


def is_routing_number(digits: str) -> bool:
    """Rule 7: nine digits with an allowed two-digit prefix (ROUTING_PREFIXES) that pass the ABA check digit."""
    if len(digits) != len(ABA_WEIGHTS):
        return False
    return int(digits[:2]) in ROUTING_PREFIXES and aba_checksum_ok(digits)


def has_routing_number(text: str) -> bool:
    """Rule 7: any stand-alone nine-digit number in text that has the form of a routing number."""
    for match in NINE_DIGITS.finditer(text):
        if is_routing_number(match.group(0)):
            return True
    return False


def treasury_line_near_form_number(text: str) -> bool:
    """Rule 8: "Department of the Treasury - Internal Revenue Service" within NEAR_CHARS characters of a form number.

    finditer returns matches left to right and they do not overlap, so both lists are sorted by start and by end.
    One pass walks them together: a form number that ends more than NEAR_CHARS before an agency line starts is too
    far left for that line and for every later one, so it is skipped for good. The first form number left is the
    leftmost one that does not end too early, so the line has a form number near it exactly when that one starts
    no more than NEAR_CHARS after the line ends. Each form number is passed over at most once, so the time grows
    with the number of matches, not with their product (comparing every line with every form number did, UP-1).
    """
    form_numbers = list(FORM_NUMBER.finditer(text))
    next_form = 0
    for agency in TREASURY_IRS.finditer(text):
        while next_form < len(form_numbers) and form_numbers[next_form].end() < agency.start() - NEAR_CHARS:
            next_form += 1
        if next_form == len(form_numbers):
            return False
        if form_numbers[next_form].start() <= agency.end() + NEAR_CHARS:
            return True
    return False


def has_tax_return_marker(text: str) -> bool:
    """Rule 8: an IRS form marker, or the agency line near a form number."""
    for marker in TAX_RETURN_MARKERS:
        if marker.search(text):
            return True
    return treasury_line_near_form_number(text)


CONTENT_CHECKS: tuple[tuple[RefusalCode, Callable[[str], bool]], ...] = (
    ("ssn", has_ssn),
    ("bank_account", has_bank_account),
    ("routing_number", has_routing_number),
    ("tax_return", has_tax_return_marker),
)


def content_refusals(texts: list[str]) -> list[RefusalCode]:
    """The refusal codes of rules 5 to 8 that any reading of any of the texts triggers, in CONTENT_CHECKS order."""
    all_readings: list[str] = []
    for text in texts:
        all_readings.extend(readings(text))
    found: list[RefusalCode] = []
    for code, check in CONTENT_CHECKS:
        if any(check(reading) for reading in all_readings):
            found.append(code)
    return found


def injection_patterns(texts: list[str]) -> tuple[str, ...]:
    """Names of the instruction-like patterns found in any of the texts (agent tripwire, in ALL_PATTERNS order)."""
    found: set[str] = set()
    for text in texts:
        found.update(detect_untrusted(text))
    return tuple(item.name for item in ALL_PATTERNS if item.name in found)


# --- the verdict ---


def verdict_message(codes: list[RefusalCode], flagged: bool) -> str:
    """The plain-language message for the codes. It is built from fixed sentences only and never quotes the file."""
    if not codes:
        if flagged:
            return f"{ACCEPTED_MESSAGE} {FLAGGED_MESSAGE}"
        return ACCEPTED_MESSAGE
    if codes[0] in GATE_MESSAGES:
        return GATE_MESSAGES[codes[0]]
    found = " and ".join(FOUND_MESSAGES[code] for code in codes)
    return (
        f"The file contains {found}, so it was not accepted. This prototype does not accept SSNs, tax returns or bank "
        f"account numbers ({PRIVACY_NOTICE})."
    )


def gate_verdict(code: RefusalCode, kind: FileKind | None = None) -> UploadVerdict:
    """A refusal at one of the gates (rules 1 to 4); the file's content was not screened."""
    return UploadVerdict(
        allowed=False,
        refusal_codes=(code,),
        file_kind=kind.name if kind is not None else None,
        message=verdict_message([code], flagged=False),
    )


def texts_to_screen(kind: FileKind, data: bytes, text: str | None) -> list[str] | None:
    """The file's texts to screen: the decoded data for CSV and plain text, plus `text` when given.

    None means a PDF or an image whose text was not read (text=None): gate 4 refuses it.
    """
    texts: list[str] = []
    if kind.magic is None:
        decoded = decode_text(data)
        if decoded is not None:
            texts.append(decoded)
    elif text is None:
        return None
    if text is not None and text not in texts:
        texts.append(text)
    return texts


def too_little_pdf_text(kind: FileKind, text: str | None) -> bool:
    """Gate 4 for a PDF: True when its text holds fewer than MIN_PDF_TEXT_CHARS visible characters.

    Whitespace and invisible characters are not counted. Such a PDF is most likely a scan whose pages are images and
    were never OCR'd (module docstring), so its content was not read.
    """
    if kind.name != "pdf" or text is None:
        return False
    visible = strip_invisible("".join(text.split()))
    return len(visible) < MIN_PDF_TEXT_CHARS


def screen_upload(filename: str, content_type: str, data: bytes, text: str | None) -> UploadVerdict:
    """Screen one uploaded file: the size, emptiness, type and readability gates, then its name and text.

    filename and content_type are what the client declared; data is the file's bytes; text is the text the route
    extracted from a PDF (its text layer plus OCR of every page without one) or an image (its OCR text), None when it
    did not read it. See the module docstring for every rule.
    """
    if len(data) > MAX_UPLOAD_BYTES or len(filename) > MAX_FILENAME_CHARS:
        return gate_verdict("too_large")
    if not data:
        return gate_verdict("empty")
    kind = kind_of(filename, content_type, data)
    if kind is None:
        return gate_verdict("file_type")
    texts = texts_to_screen(kind, data, text)
    if texts is None:
        return gate_verdict("unreadable", kind)
    if any(len(item) > MAX_TEXT_CHARS for item in texts):
        return gate_verdict("too_large", kind)
    if too_little_pdf_text(kind, text):
        return gate_verdict("unreadable", kind)
    name_as_words = code_as_words(filename)
    codes = content_refusals([filename, name_as_words, *texts])
    patterns = injection_patterns([name_as_words, *texts])
    return UploadVerdict(
        allowed=not codes,
        refusal_codes=tuple(codes),
        file_kind=kind.name,
        injection_flagged=bool(patterns),
        injection_patterns=patterns,
        message=verdict_message(codes, flagged=bool(patterns)),
    )
