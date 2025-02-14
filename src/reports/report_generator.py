import pandas as pd
from fpdf import FPDF
import xlsxwriter
from datetime import datetime
import plotly.io as pio
import tempfile
from pathlib import Path
import logging
from typing import Dict, List
import base64
import io

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, claims_data: pd.DataFrame, analytics_data: Dict = None):
        """Initialize report generator"""
        self.claims_data = claims_data
        self.analytics_data = analytics_data or {}
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)

    def generate_pdf_report(self, report_type: str, metrics: List[str]) -> bytes:
        """Generate PDF report"""
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f'Insurance Claims {report_type}', ln=True, align='C')
            pdf.ln(10)
            
            # Add timestamp
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
            pdf.ln(10)
            
            # Add metrics
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Key Metrics', ln=True)
            pdf.ln(5)
            
            pdf.set_font('Arial', '', 10)
            for metric in metrics:
                value = self._get_metric_value(metric)
                pdf.cell(0, 10, f'{metric}: {value}', ln=True)
            
            # Add visualizations
            if 'Trends' in metrics:
                self._add_visualization_to_pdf(pdf, 'trends')
            if 'Fraud Risk' in metrics:
                self._add_visualization_to_pdf(pdf, 'fraud')
            
            return pdf.output(dest='S').encode('latin1')
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise

    def generate_excel_report(self, report_type: str, metrics: List[str]) -> bytes:
        """Generate Excel report"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_data = self._get_summary_data(metrics)
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed data
                if 'Claims Count' in metrics:
                    self._add_claims_sheet(writer)
                if 'Fraud Risk' in metrics:
                    self._add_fraud_sheet(writer)
                if 'Cost Distribution' in metrics:
                    self._add_cost_sheet(writer)
                if 'Trends' in metrics:
                    self._add_trends_sheet(writer)
                
                # Format sheets
                self._format_excel_sheets(writer)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Excel generation failed: {e}")
            raise

    def _get_metric_value(self, metric: str) -> str:
        """Get formatted metric value"""
        try:
            if metric == 'Claims Count':
                return str(len(self.claims_data))
            elif metric == 'Total Amount':
                return f"${self.claims_data['total_amount'].sum():,.2f}"
            elif metric == 'Average Processing Time':
                return f"{self.claims_data['processing_time'].mean():.1f} days"
            elif metric == 'Fraud Risk':
                return f"{self.claims_data['fraud_risk'].mean():.1f}%"
            else:
                return "N/A"
        except Exception as e:
            logger.error(f"Failed to get metric value: {e}")
            return "Error"

    def _add_visualization_to_pdf(self, pdf: FPDF, viz_type: str):
        """Add visualization to PDF"""
        try:
            # Create visualization
            if viz_type == 'trends':
                fig = self._create_trends_plot()
            elif viz_type == 'fraud':
                fig = self._create_fraud_plot()
            else:
                return
            
            # Save plot to temporary file
            temp_file = self.temp_dir / f"{viz_type}_plot.png"
            pio.write_image(fig, str(temp_file))
            
            # Add to PDF
            pdf.add_page()
            pdf.image(str(temp_file), x=10, y=10, w=190)
            
            # Cleanup
            temp_file.unlink()
            
        except Exception as e:
            logger.error(f"Failed to add visualization: {e}")

    def _get_summary_data(self, metrics: List[str]) -> pd.DataFrame:
        """Get summary data for Excel report"""
        summary_data = []
        
        for metric in metrics:
            summary_data.append({
                'Metric': metric,
                'Value': self._get_metric_value(metric)
            })
        
        return pd.DataFrame(summary_data)

    def _add_claims_sheet(self, writer: pd.ExcelWriter):
        """Add claims data sheet"""
        claims_summary = self.claims_data.groupby('month').agg({
            'claim_id': 'count',
            'total_amount': 'sum',
            'fraud_risk': 'mean'
        }).reset_index()
        
        claims_summary.to_excel(writer, sheet_name='Claims_Summary', index=False)

    def _format_excel_sheets(self, writer: pd.ExcelWriter):
        """Format Excel sheets"""
        workbook = writer.book
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4F81BD',
            'font_color': 'white'
        })
        
        currency_format = workbook.add_format({'num_format': '$#,##0.00'})
        percent_format = workbook.add_format({'num_format': '0.0%'})
        
        # Apply formats to each sheet
        for worksheet in writer.sheets.values():
            worksheet.set_column('A:Z', 15)  # Set column width
            worksheet.set_row(0, None, header_format)  # Format header row 