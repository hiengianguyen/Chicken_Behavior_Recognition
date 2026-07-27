from pathlib import Path
import pandas as pd


class BehaviorExporter:

    def __init__(self, records):

        self.records = records

    def export_csv(self, output_path: str):

        output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([record.to_dict() for record in self.records])

        df.to_csv(output_path, index=False)

        print()

        print("=" * 50)
        print("Behavior CSV Saved")
        print(f"Rows : {len(df)}")
        print(f"Path : {output_path}")
        print("=" * 50)
