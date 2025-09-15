"""
Expense Estimation Module for Seoul Market Risk ML System
소상공인 지출 평균 75.44867193% 기준으로 지출 추정
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ExpenseEstimator:
    def __init__(self, expense_ratio: float = 0.7544867193):
        self.expense_ratio = expense_ratio

    def add_expense_column_to_csv(self, csv_file_path: str) -> str:
        df = pd.read_csv(csv_file_path)
        df['추정지출금액'] = (df['당월_매출_금액'] * self.expense_ratio).round().astype(int)
        df.to_csv(csv_file_path, index=False)
        return csv_file_path

    def process_all_csv_files(self, data_directory: str = "data/raw") -> List[str]:
        data_path = Path(data_directory)
        csv_files = list(data_path.glob("*상권분석서비스*.csv"))

        processed_files = []
        for csv_file in csv_files:
            try:
                processed_path = self.add_expense_column_to_csv(str(csv_file))
                processed_files.append(processed_path)
                logger.info(f"✅ Processed: {csv_file.name}")
            except Exception as e:
                logger.error(f"❌ Failed: {csv_file.name} - {e}")

        return processed_files