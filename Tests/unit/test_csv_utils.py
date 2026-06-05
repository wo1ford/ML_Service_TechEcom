import csv
import io
from io import StringIO


class TestCSVParsing:

    def test_parse_valid(self):
        csv_text = "orders,couriers\n100,1\n300,2\n"
        reader = csv.DictReader(StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 2
    
    def test_parse_extra_columns(self):
        """Лишние колонки игнорируются."""
        csv_text = "orders,couriers,extra\n100,1,x\n300,2,y\n"
        reader = csv.DictReader(StringIO(csv_text))
        rows = list(reader)
        assert rows[0]["orders"] == "100"
        assert rows[0]["extra"] == "x"
    
    def test_parse_empty(self):
        csv_text = "orders,couriers\n"
        reader = csv.DictReader(StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 0
    
    def test_parse_different_delimiter(self):
        """Точка с запятой."""
        csv_text = "orders;couriers\n100;1\n300;2\n"
        reader = csv.DictReader(StringIO(csv_text), delimiter=";")
        rows = list(reader)
        assert rows[0]["orders"] == "100"
    
    def test_write_result(self):
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["orders", "couriers", "deficit"])
        writer.writerow([100, 1, 0])
        
        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)
        
        assert reader.fieldnames == ["orders", "couriers", "deficit"]
        assert rows[0]["deficit"] == "0"
    
    def test_roundtrip(self):
        """Полный цикл запись => чтение"""
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["orders", "couriers", "deficit"])
        
        data = [(100, 1, 0), (300, 2, 1), (50, 5, 0)]
        for orders, couriers, deficit in data:
            writer.writerow([orders, couriers, deficit])
        
        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)
        
        assert len(rows) == 3
        for i, row in enumerate(rows):
            assert int(row["orders"]) == data[i][0]
            assert int(row["couriers"]) == data[i][1]
            assert int(row["deficit"]) == data[i][2]