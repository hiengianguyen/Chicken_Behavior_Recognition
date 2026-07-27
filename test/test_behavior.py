from src.behavior.extractor import BehaviorExtractor
from src.behavior.exporter import BehaviorExporter

extractor = BehaviorExtractor(
    "datasets/trajectory/b_6.csv"
)

records = extractor.extract()

exporter = BehaviorExporter(records)

exporter.export_csv(
    "datasets/behavior/b_6.csv"
)