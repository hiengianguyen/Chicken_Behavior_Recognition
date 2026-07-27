from pathlib import Path
import pandas as pd

from .manager import TrajectoryManager


class TrajectoryExporter:
    """
    Xuất dữ liệu trajectory ra các định dạng khác nhau.
    """

    def __init__(self, manager: TrajectoryManager):
        self.manager = manager

    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển toàn bộ trajectory thành DataFrame.
        """

        rows = []

        trajectories = self.manager.get_all()

        for trajectory in trajectories.values():

            for record in trajectory.records:

                rows.append(record.to_dict())

        df = pd.DataFrame(rows)

        if not df.empty:
            df.sort_values(by=["frame", "track_id"], inplace=True)

            df.reset_index(drop=True, inplace=True)

        return df

    def export_csv(self, output_path: str):

        output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()

        df.to_csv(output_path, index=False)

        print(f"[TrajectoryExporter] Saved -> {output_path}")

    def export_excel(self, output_path: str):

        output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()

        df.to_excel(output_path, index=False)

        print(f"[TrajectoryExporter] Saved -> {output_path}")
