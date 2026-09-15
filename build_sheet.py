"""Builds the Google Sheets companion workbook.

Deliberately NOT passed through recalc.py: every price formula is
GOOGLEFINANCE, which only exists inside Google Sheets. LibreOffice cannot
evaluate it and would bake #NAME? into every cell permanently. The file is
built to be uploaded to Google Sheets, where the formulas come alive.
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import sys
sys.path.insert(0, ".")
from tickers import TICKERS

ARIAL = "Arial"
INK = "FF1B2124"
MUTED = "FF5B6B70"
BRASS = "FF7A5415"
HEAD_FILL = PatternFill("solid", fgColor="FFE7EBE7")
INPUT_FILL = PatternFill("solid", fgColor="FFFFF3C4")   # yellow = you type here
TITLE_FILL = PatternFill("solid", fgColor="FFF6E8CD")
thin = Side(style="thin", color="FFD2D8D2")
BORDER = Border(bottom=thin)

wb = Workbook()
ws = wb.active
ws.title = "Watchlist"

# ---------------------------------------------------------------- settings
ws["A1"] = "MIDCAP REVERSAL DESK — live sheet"
ws["A1"].font = Font(name=ARIAL, size=15, bold=True, color=INK)
ws.merge_cells("A1:E1")
ws["A1"].fill = TITLE_FILL

ws["A2"] = "Total risk capital (₹)"
ws["B2"] = 100000
ws["A3"] = "Risk per trade — RPT (total ÷ 50)"
ws["B3"] = "=B2/50"
ws["A4"] = "Stop distance (ATR ×)"
ws["B4"] = 1.5

for r in (2, 3, 4):
    ws[f"A{r}"].font = Font(name=ARIAL, size=10, color=MUTED)
ws["B2"].font = Font(name=ARIAL, size=11, bold=True, color="FF0000FF")   # input
ws["B4"].font = Font(name=ARIAL, size=11, bold=True, color="FF0000FF")   # input
ws["B3"].font = Font(name=ARIAL, size=11, bold=True, color=BRASS)        # formula
ws["B2"].fill = INPUT_FILL
ws["B4"].fill = INPUT_FILL
ws["B2"].number_format = "#,##0"
ws["B3"].number_format = "#,##0"
ws["B4"].number_format = "0.0"

ws["D2"] = "Blue = you type it.  Yellow = you type it.  Everything else is a formula."
ws["D2"].font = Font(name=ARIAL, size=9, italic=True, color=MUTED)
ws["D3"] = "Prices are live via GOOGLEFINANCE. RSI and ATR are simple-average"
ws["D4"] = "approximations — see the Read me tab before you trade off them."
for c in ("D3", "D4"):
    ws[c].font = Font(name=ARIAL, size=9, italic=True, color=MUTED)

# ---------------------------------------------------------------- header
HEAD_ROW = 6
HEADERS = [
    ("Symbol", 13), ("Company", 30), ("Wt %", 7), ("Price ₹", 11),
    ("RSI(14)~", 10), ("ATR(14)~", 10), ("Resistance ₹", 13), ("Divergence?", 12),
    ("Stop ₹", 11), ("Risk/share ₹", 12), ("Qty", 9), ("Deploy ₹", 12),
    ("To break", 10), ("State", 14), ("Note", 26),
]
for i, (label, width) in enumerate(HEADERS, start=1):
    c = ws.cell(row=HEAD_ROW, column=i, value=label)
    c.font = Font(name=ARIAL, size=10, bold=True, color=INK)
    c.fill = HEAD_FILL
    c.border = BORDER
    c.alignment = Alignment(horizontal="left" if i <= 2 else "right", wrap_text=False)
    ws.column_dimensions[get_column_letter(i)].width = width

# ---------------------------------------------------------------- rows
RSI_F = (
    '=IFERROR(LET('
    'r, GOOGLEFINANCE("NSE:"&$A{n},"all",TODAY()-120,TODAY()),'
    'c, FILTER(INDEX(r,,5),ISNUMBER(INDEX(r,,5))),'
    'k, ROWS(c),'
    'd, INDEX(c,SEQUENCE(14,1,k-13,1))-INDEX(c,SEQUENCE(14,1,k-14,1)),'
    'g, SUMPRODUCT(d,--(d>0)),'
    'l, -SUMPRODUCT(d,--(d<0)),'
    'IF(l=0,100,ROUND(100-100/(1+g/l),1))),"")'
)
ATR_F = (
    '=IFERROR(LET('
    'r, GOOGLEFINANCE("NSE:"&$A{n},"all",TODAY()-120,TODAY()),'
    'h, FILTER(INDEX(r,,3),ISNUMBER(INDEX(r,,3))),'
    'w, FILTER(INDEX(r,,4),ISNUMBER(INDEX(r,,4))),'
    'c, FILTER(INDEX(r,,5),ISNUMBER(INDEX(r,,5))),'
    'k, ROWS(c),'
    'hi, INDEX(h,SEQUENCE(14,1,k-13,1)),'
    'lo, INDEX(w,SEQUENCE(14,1,k-13,1)),'
    'cp, INDEX(c,SEQUENCE(14,1,k-14,1)),'
    'm, ARRAYFORMULA(IF(hi-lo>ABS(hi-cp),hi-lo,ABS(hi-cp))),'
    't, ARRAYFORMULA(IF(m>ABS(lo-cp),m,ABS(lo-cp))),'
    'ROUND(AVERAGE(t),2)),"")'
)
STATE_F = (
    '=IF($G{n}="","set resistance",'
    'IF(AND($H{n}="Yes",$D{n}>=$G{n}),"TRIGGERED",'
    'IF($H{n}="Yes","ARMED",'
    'IF($E{n}="","",'
    'IF($E{n}<30,"Oversold",IF($E{n}>70,"Overbought","Watching"))))))'
)

start = HEAD_ROW + 1
for i, (sym, name, weight) in enumerate(TICKERS):
    n = start + i
    ws.cell(row=n, column=1, value=sym).font = Font(name=ARIAL, size=10, bold=True, color=INK)
    ws.cell(row=n, column=2, value=name).font = Font(name=ARIAL, size=10, color=MUTED)
    ws.cell(row=n, column=3, value=weight).number_format = "0.00"
    ws.cell(row=n, column=4, value=f'=IFERROR(GOOGLEFINANCE("NSE:"&$A{n},"price"),"")')
    ws.cell(row=n, column=5, value=RSI_F.format(n=n))
    ws.cell(row=n, column=6, value=ATR_F.format(n=n))
    ws.cell(row=n, column=7, value=None)                       # resistance: manual
    ws.cell(row=n, column=8, value=None)                       # divergence: manual
    ws.cell(row=n, column=9,  value=f'=IFERROR(IF($G{n}="","",$G{n}-$J{n}),"")')
    ws.cell(row=n, column=10, value=f'=IFERROR(IF($F{n}="","",ROUND($F{n}*$B$4,2)),"")')
    ws.cell(row=n, column=11, value=f'=IFERROR(IF($J{n}="","",FLOOR($B$3/$J{n})),"")')
    ws.cell(row=n, column=12, value=f'=IFERROR(IF($K{n}="","",ROUND($K{n}*$G{n},0)),"")')
    ws.cell(row=n, column=13, value=f'=IFERROR(IF(OR($G{n}="",$D{n}=""),"",($G{n}-$D{n})/$D{n}),"")')
    ws.cell(row=n, column=14, value=STATE_F.format(n=n))
    ws.cell(row=n, column=15, value=None)                      # note: manual

    for col in range(1, 16):
        cell = ws.cell(row=n, column=col)
        cell.border = BORDER
        if col > 2:
            cell.alignment = Alignment(horizontal="right")
        if cell.font is None or cell.font.name != ARIAL:
            cell.font = Font(name=ARIAL, size=10, color=INK)

    for col in (7, 8, 15):                    # the three columns you fill in
        ws.cell(row=n, column=col).fill = INPUT_FILL
        ws.cell(row=n, column=col).font = Font(name=ARIAL, size=10, bold=True, color="FF0000FF")
    ws.cell(row=n, column=15).alignment = Alignment(horizontal="left")

    for col, fmt in ((4, "#,##0.00"), (5, "0.0"), (6, "#,##0.00"), (7, "#,##0.00"),
                     (9, "#,##0.00"), (10, "#,##0.00"), (11, "#,##0"),
                     (12, "#,##0"), (13, "0.0%")):
        ws.cell(row=n, column=col).number_format = fmt

last = start + len(TICKERS) - 1
dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
ws.add_data_validation(dv)
dv.add(f"H{start}:H{last}")

ws.freeze_panes = f"C{start}"
ws.auto_filter.ref = f"A{HEAD_ROW}:O{last}"

# ---------------------------------------------------------------- read me
rm = wb.create_sheet("Read me")
rm.column_dimensions["A"].width = 100
LINES = [
    ("MIDCAP REVERSAL DESK — the sheet version", "h1"),
    ("", ""),
    ("How to bring it to life", "h2"),
    ("1. Go to sheets.google.com, then File ▸ Import ▸ Upload, and drop this file in.", ""),
    ("2. Choose 'Replace spreadsheet' and import. Prices start filling within seconds.", ""),
    ("3. Nothing else to set up. Google recalculates it whenever you open it.", ""),
    ("", ""),
    ("Opening it in Excel will NOT work: GOOGLEFINANCE, LET and SEQUENCE are Google Sheets", ""),
    ("functions. In Excel every price cell shows #NAME?. That is expected.", ""),
    ("", ""),
    ("What you fill in (the yellow columns)", "h2"),
    ("Resistance ₹  — the previous major swing high price has to break. Read it off your chart.", ""),
    ("Divergence?   — Yes once you see price making a lower bottom while RSI makes a higher bottom.", ""),
    ("Note          — anything you want to remember about the setup.", ""),
    ("", ""),
    ("What the sheet works out for you", "h2"),
    ("Risk per share = ATR × the multiplier in B4.  Stop = resistance − risk per share.", ""),
    ("Qty = RPT ÷ risk per share, where RPT = total risk capital ÷ 50.", ""),
    ("State reads TRIGGERED once you have marked the divergence and price closes at or above", ""),
    ("your resistance; ARMED while it is still below.", ""),
    ("", ""),
    ("An honest caveat about RSI and ATR here", "h2"),
    ("The RSI(14)~ and ATR(14)~ columns use plain 14-day averages, because Wilder's smoothing", ""),
    ("cannot be expressed cleanly in a single spreadsheet formula. They track the real thing", ""),
    ("closely but will not match your chart to the decimal — the tilde in the header is the", ""),
    ("reminder. The GitHub scanner that powers the web dashboard uses true Wilder RSI and ATR,", ""),
    ("so treat this sheet as the quick live view and the dashboard as the precise one.", ""),
    ("", ""),
    ("Also worth knowing", "h2"),
    ("GOOGLEFINANCE quotes are delayed (typically up to 20 minutes) and a few very recently", ""),
    ("listed stocks may return nothing at all — those cells stay blank rather than guess.", ""),
    ("Adding a stock: type its NSE symbol in column A of a new row and copy the formulas down", ""),
    ("from the row above.", ""),
    ("", ""),
    ("This automates a strategy you defined. It is not investment advice — confirm every level", ""),
    ("on your own chart before placing an order.", ""),
]
for i, (text, kind) in enumerate(LINES, start=1):
    c = rm.cell(row=i, column=1, value=text)
    if kind == "h1":
        c.font = Font(name=ARIAL, size=14, bold=True, color=INK)
        c.fill = TITLE_FILL
    elif kind == "h2":
        c.font = Font(name=ARIAL, size=11, bold=True, color=BRASS)
    else:
        c.font = Font(name=ARIAL, size=10, color=INK)

wb.save("Midcap-Reversal-Desk.xlsx")
print(f"wrote Midcap-Reversal-Desk.xlsx with {len(TICKERS)} stocks, rows {start}-{last}")
