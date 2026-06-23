import pandas as pd
import os
from typing import List, Dict, Any

class InstitutionalReports:
    def __init__(self, output_dir: str = "reports/institutional"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_divergence_report(self, data: List[Dict[str, Any]], filename: str = "trade_divergence.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def export_deployment_readiness(self, data: List[Dict[str, Any]], filename: str = "deployment_readiness.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
