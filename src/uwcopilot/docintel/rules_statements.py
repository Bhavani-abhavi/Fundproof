"""Rules extraction of one SYNTHETIC bank statement (credit packets) into a cited BankStatement.

Header fields are label/value lines (rules_common.find_value): Account holder, Account number, Statement period
("YYYY-MM-DD to YYYY-MM-DD"; both dates cite the same words), Opening balance, Total deposits, Total withdrawals,
Closing balance. The bank name is the letterhead: the first line of the statement.

Transactions: the table header is the first line holding at least three of Date / Description / Withdrawals /
Deposits / Balance. Every later line that starts with a date (within its first two words) is a transaction row;
lines that do not (memo lines and their continuations, "Number of transactions") are skipped. In a row:
- the table amounts (printed with two decimals) are found by pattern; the LAST one is the running balance and the
  one before it is the transaction amount; a row with a single amount is a balance only when that amount sits
  nearer the Balance column than the amount columns;
- the description is the words between the date and the first amount;
- deposit or withdrawal: the amount goes to whichever of the Withdrawals / Deposits columns has its right edge
  (amounts are right-aligned) nearest the amount's right edge. When the header titles cannot be read (OCR), the
  running balance decides: previous balance + amount = balance means a deposit, previous balance - amount = balance
  a withdrawal (previous balance = the row above, or the opening balance for the first row). If neither holds,
  the amount is left out rather than guessed.
"""

from uwcopilot.docintel.classify import Segment
from uwcopilot.docintel.layout import Word
from uwcopilot.docintel.parsing import fix_ocr_digits, parse_date, parse_money, parse_text, strip_edges
from uwcopilot.docintel.rules_common import (
    LineRef,
    amount_words,
    cite_words,
    document_lines,
    find_value,
    header_index,
    title_box,
    value_after_label,
    words_text,
)
from uwcopilot.docintel.schemas import BankStatement, Cited, Transaction

TABLE_TITLES = ("Date", "Description", "Withdrawals", "Deposits", "Balance")
BALANCE_TOLERANCE = 0.01
ACCOUNT_DIGITS = (10, 12)


def parse_account_number(text: str) -> str | None:
    """The account number when the text is one run of 10-12 digits (after the OCR digit fix), else None."""
    fixed = fix_ocr_digits(strip_edges(text))
    low, high = ACCOUNT_DIGITS
    if fixed.isdigit() and low <= len(fixed) <= high:
        return fixed
    return None


def period_values(refs: list[LineRef], doc_id: str) -> tuple[Cited, Cited]:
    """(period_start, period_end) from the "Statement period A to B" line."""
    for ref in refs:
        words = value_after_label(ref.line, "Statement period", ref.fuzzy)
        if not words:
            continue
        dates = [(word, parse_date(word.text)) for word in words]
        found = [(word, value) for word, value in dates if value is not None]
        if len(found) >= 2:
            return cite_words(ref, words, found[0][1], doc_id), cite_words(ref, words, found[-1][1], doc_id)
    return Cited(), Cited()


def letterhead(refs: list[LineRef], doc_id: str) -> Cited:
    """The bank name: the first line of the statement."""
    if not refs:
        return Cited()
    return cite_words(refs[0], refs[0].words, parse_text(refs[0].text), doc_id)


def amount_columns(header: LineRef | None) -> dict[str, float]:
    """Right edge of the Withdrawals and Deposits titles on the header line (the titles that were found)."""
    columns: dict[str, float] = {}
    if header is None:
        return columns
    for name, title in (("withdrawal", "Withdrawals"), ("deposit", "Deposits"), ("balance", "Balance")):
        box = title_box(header, title)
        if box is not None:
            columns[name] = box[1]
    return columns


def row_date(ref: LineRef) -> tuple[Word | None, int]:
    """The date word of a transaction row (within its first two words) and its index, or (None, -1)."""
    for index, word in enumerate(ref.words[:2]):
        if parse_date(word.text) is not None:
            return word, index
    return None, -1


def side_by_column(amount: Word, columns: dict[str, float]) -> str | None:
    """"deposit" or "withdrawal" by the nearest amount column, or None when either column is unknown."""
    if "deposit" not in columns or "withdrawal" not in columns:
        return None
    to_deposit = abs(columns["deposit"] - amount.x1)
    to_withdrawal = abs(columns["withdrawal"] - amount.x1)
    return "deposit" if to_deposit < to_withdrawal else "withdrawal"


def side_by_balance(amount: float, balance: float | None, previous: float | None) -> str | None:
    """"deposit" or "withdrawal" from the running balance arithmetic, or None when it does not decide."""
    if balance is None or previous is None:
        return None
    if abs(previous + amount - balance) <= BALANCE_TOLERANCE:
        return "deposit"
    if abs(previous - amount - balance) <= BALANCE_TOLERANCE:
        return "withdrawal"
    return None


def split_amounts(amounts: list[Word], columns: dict[str, float]) -> tuple[Word | None, Word | None]:
    """(transaction amount word, balance word) of a row (see the module docstring)."""
    if len(amounts) >= 2:
        return amounts[-2], amounts[-1]
    if len(amounts) == 1:
        only = amounts[0]
        balance_x = columns.get("balance")
        others = [columns[name] for name in ("deposit", "withdrawal") if name in columns]
        if balance_x is not None and others and abs(balance_x - only.x1) < min(abs(x - only.x1) for x in others):
            return None, only
        return only, None
    return None, None


def parse_transaction(
    ref: LineRef, columns: dict[str, float], previous: float | None, doc_id: str
) -> Transaction | None:
    """One transaction row, or None when the line is not a row (see the module docstring)."""
    date_word, date_index = row_date(ref)
    if date_word is None:
        return None
    rest = ref.words[date_index + 1 :]
    amounts = amount_words(rest)
    amount_word, balance_word = split_amounts(amounts, columns)
    first_amount = amounts[0] if amounts else None
    description = [word for word in rest if first_amount is None or word.x1 < first_amount.x0]
    transaction = Transaction(posted_date=cite_words(ref, [date_word], parse_date(date_word.text), doc_id))
    if description:
        transaction.description = cite_words(ref, description, parse_text(words_text(description)), doc_id)
    balance = None if balance_word is None else parse_money(balance_word.text)
    if balance_word is not None and balance is not None:
        transaction.balance = cite_words(ref, [balance_word], balance, doc_id)
    if amount_word is None:
        return transaction
    amount = parse_money(amount_word.text)
    if amount is None:
        return transaction
    side = side_by_column(amount_word, columns) or side_by_balance(amount, balance, previous)
    if side == "deposit":
        transaction.deposit = cite_words(ref, [amount_word], amount, doc_id)
    elif side == "withdrawal":
        transaction.withdrawal = cite_words(ref, [amount_word], amount, doc_id)
    return transaction


def transactions(refs: list[LineRef], opening: float | None, doc_id: str) -> list[Transaction]:
    """Every transaction row after the table header (see the module docstring)."""
    start = header_index(refs, TABLE_TITLES, 3)
    if start is None:
        return []
    columns = amount_columns(refs[start])
    rows = []
    previous = opening
    for ref in refs[start + 1 :]:
        row = parse_transaction(ref, columns, previous, doc_id)
        if row is None:
            continue
        rows.append(row)
        previous = row.balance.value if row.balance.found else None
    return rows


def extract_statement(segment: Segment) -> BankStatement:
    """One bank statement document."""
    refs = document_lines(segment)
    doc_id = segment.doc_id
    start, end = period_values(refs, doc_id)
    opening = find_value(refs, ("Opening balance",), parse_money, doc_id)
    return BankStatement(
        doc_id=doc_id,
        bank_name=letterhead(refs, doc_id),
        account_holder=find_value(refs, ("Account holder",), parse_text, doc_id),
        account_number=find_value(refs, ("Account number",), parse_account_number, doc_id),
        period_start=start,
        period_end=end,
        opening_balance=opening,
        total_deposits=find_value(refs, ("Total deposits",), parse_money, doc_id),
        total_withdrawals=find_value(refs, ("Total withdrawals",), parse_money, doc_id),
        closing_balance=find_value(refs, ("Closing balance",), parse_money, doc_id),
        transactions=transactions(refs, opening.value, doc_id),
    )
