from zipfile import ZipFile

from evolab.contracts.common import RuntimePolicy
from evolab.contracts.tools import ToolCall
from evolab.tools.runtime import ToolRegistry, ToolRuntime
from evolab.tools.tables import register_table_tools


def _runtime() -> ToolRuntime:
    registry = ToolRegistry()
    register_table_tools(registry)
    runtime = ToolRuntime(registry)
    tools = [
        "inspect_table",
        "read_table_slice",
        "inspect_excel_workbook",
        "read_excel_sheet",
        "detect_table_header",
        "normalize_table",
        "profile_table",
    ]
    runtime.prepare(required_tools=tools, allowed_tools=tools, policy=RuntimePolicy())
    return runtime


def test_csv_markdown_and_table_structure_tools_smoke(tmp_path):
    csv_path = tmp_path / "table.csv"
    csv_path.write_text("name,count,sequence\nalpha,2,ACGTACGT\nbeta,3,GGGGTTTT\n", encoding="utf-8")
    md_path = tmp_path / "table.md"
    md_path.write_text("| name | count |\n| --- | --- |\n| alpha | 2 |\n", encoding="utf-8")
    runtime = _runtime()

    inspected = runtime.execute(ToolCall(call_id="inspect", name="inspect_table", arguments={"path": str(csv_path)}))
    sliced = runtime.execute(ToolCall(call_id="slice", name="read_table_slice", arguments={"path": str(csv_path), "start_row": 1}))
    header = runtime.execute(
        ToolCall(call_id="header", name="detect_table_header", arguments={"rows": [["name", "count"], ["alpha", "2"]]})
    )
    normalized = runtime.execute(
        ToolCall(call_id="normalize", name="normalize_table", arguments={"rows": [["Name", " Count "], [" alpha ", "2"]]})
    )
    profiled = runtime.execute(ToolCall(call_id="profile", name="profile_table", arguments={"path": str(csv_path)}))
    markdown = runtime.execute(ToolCall(call_id="md", name="inspect_table", arguments={"path": str(md_path)}))

    assert inspected.status == "ok"
    assert inspected.metadata["headers"] == ["name", "count", "sequence"]
    assert sliced.status == "ok"
    assert sliced.metadata["rows"][0] == ["alpha", "2", "ACGTACGT"]
    assert header.status == "ok"
    assert header.metadata["header_row_index"] == 0
    assert normalized.status == "ok"
    assert normalized.metadata["headers"] == ["name", "count"]
    assert profiled.status == "ok"
    assert profiled.metadata["column_profiles"][2]["dna_like_fraction"] == 1.0
    assert markdown.status == "ok"
    assert markdown.metadata["headers"] == ["name", "count"]


def test_plain_text_markdown_table_block_can_be_read_by_caption(tmp_path):
    md_path = tmp_path / "supplement.md"
    md_path.write_text(
        "\n".join(
            [
                "Tables",
                "Table S1 Promoter activity",
                "Names",
                "Promoter sequences",
                "FI/OD600",
                "J23119",
                "TTGACAGCAATCTCAATCCTAGGTATAAT",
                "51747",
                "S1",
                "AAATCAATTAATTCTTGACAGTTAGCTCATTCCTAGGTATA",
                "ATGCGAGCA",
                "449926",
                "Table S2 Other table",
                "Name",
                "Value",
                "D1",
                "123",
            ]
        ),
        encoding="utf-8",
    )
    runtime = _runtime()

    inspected = runtime.execute(
        ToolCall(
            call_id="inspect",
            name="inspect_table",
            arguments={"path": str(md_path), "table_caption": "Table S1"},
        )
    )
    sliced = runtime.execute(
        ToolCall(
            call_id="slice",
            name="read_table_slice",
            arguments={"path": str(md_path), "table_caption": "Table S1", "start_row": 1},
        )
    )

    assert inspected.status == "ok"
    assert inspected.metadata["headers"] == ["names", "promoter_sequences", "fi_od600"]
    assert sliced.status == "ok"
    assert sliced.metadata["rows"] == [
        ["J23119", "TTGACAGCAATCTCAATCCTAGGTATAAT", "51747"],
        ["S1", "AAATCAATTAATTCTTGACAGTTAGCTCATTCCTAGGTATAATGCGAGCA", "449926"],
    ]
    assert sliced.metadata["plain_text_table_block"]["caption"] == "Table S1 Promoter activity"


def test_plain_text_markdown_table_caption_tolerates_source_line_slice_indexes(tmp_path):
    md_path = tmp_path / "supplement.md"
    md_path.write_text(
        "\n".join(
            [
                "intro",
                "Table S1 Promoter activity",
                "Names",
                "Promoter sequences",
                "FI/OD600",
                "J23119",
                "TTGACAGCAATCTCAATCCTAGGTATAAT",
                "51747",
                "S1",
                "AAATCAATTAATTCTTGACAGTTAGCTCATTCCTAGGTATA",
                "ATGCGAGCA",
                "449926",
            ]
        ),
        encoding="utf-8",
    )
    runtime = _runtime()

    sliced = runtime.execute(
        ToolCall(
            call_id="slice",
            name="read_table_slice",
            arguments={
                "path": str(md_path),
                "table_caption": "Table S1",
                "start_row": 2,
                "end_row": 12,
            },
        )
    )

    assert sliced.status == "ok"
    assert sliced.metadata["start_row"] == 0
    assert sliced.metadata["end_row"] == 3
    assert sliced.metadata["rows"][1][0] == "J23119"
    assert sliced.metadata["rows"][2][0] == "S1"
    assert "interpreted start_row/end_row as source line indexes" in sliced.metadata["warnings"]


def test_minimal_xlsx_tools_smoke_without_openpyxl(tmp_path):
    xlsx_path = tmp_path / "workbook.xlsx"
    _write_minimal_xlsx(xlsx_path)
    runtime = _runtime()

    workbook = runtime.execute(ToolCall(call_id="workbook", name="inspect_excel_workbook", arguments={"path": str(xlsx_path)}))
    sheet = runtime.execute(
        ToolCall(call_id="sheet", name="read_excel_sheet", arguments={"path": str(xlsx_path), "sheet_name": "Sheet1"})
    )
    inspected = runtime.execute(ToolCall(call_id="inspect-xlsx", name="inspect_table", arguments={"path": str(xlsx_path)}))

    assert workbook.status == "ok"
    assert workbook.metadata["sheets"][0]["sheet_name"] == "Sheet1"
    assert sheet.status == "ok"
    assert sheet.metadata["rows"][1] == ["alpha", "2"]
    assert inspected.status == "ok"
    assert inspected.metadata["headers"] == ["name", "count"]


def _write_minimal_xlsx(path):
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "")
        archive.writestr(
            "xl/workbook.xml",
            """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                <Relationship Id="rId1" Target="worksheets/sheet1.xml"
                 Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
                </Relationships>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                <sheetData>
                <row r="1"><c r="A1" t="inlineStr"><is><t>name</t></is></c><c r="B1" t="inlineStr"><is><t>count</t></is></c></row>
                <row r="2"><c r="A2" t="inlineStr"><is><t>alpha</t></is></c><c r="B2"><v>2</v></c></row>
                </sheetData></worksheet>""",
        )
