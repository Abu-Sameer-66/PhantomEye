import os
from datetime import datetime
from fpdf import FPDF
import numpy as np
import cv2
import tempfile


FONT_PATH = None  # Use built-in fonts


class PhantomEyeReport(FPDF):

    def header(self):
        self.set_fill_color(10, 10, 10)
        self.rect(0, 0, 210, 297, 'F')
        self.set_fill_color(0, 30, 15)
        self.rect(0, 0, 210, 18, 'F')
        self.set_font('Courier', 'B', 14)
        self.set_text_color(0, 255, 136)
        self.cell(0, 12, 'PHANTOMEYE INTELLIGENCE REPORT', align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Courier', '', 8)
        self.set_text_color(0, 170, 85)
        self.cell(0, 6, f'GENERATED: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  |  CLASSIFICATION: CONFIDENTIAL', align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Courier', '', 7)
        self.set_text_color(0, 100, 50)
        self.cell(0, 10, f'PhantomEye AI Surveillance System  |  Page {self.page_no()}  |  CONFIDENTIAL', align='C')

    def section_title(self, title: str):
        self.set_fill_color(0, 40, 20)
        self.set_text_color(0, 255, 136)
        self.set_font('Courier', 'B', 10)
        self.cell(0, 8, f'  {title}', fill=True, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def key_value(self, key: str, value: str):
        self.set_font('Courier', 'B', 9)
        self.set_text_color(0, 200, 100)
        self.cell(60, 6, f'  {key}:', new_x='RIGHT', new_y='LAST')
        self.set_font('Courier', '', 9)
        self.set_text_color(200, 255, 200)
        self.cell(0, 6, str(value), new_x='LMARGIN', new_y='NEXT')

    def add_image_section(self, title: str, img_array: np.ndarray):
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, img_array)
            self.section_title(title)
            page_w = self.w - 2 * self.l_margin
            self.image(tmp_path, x=self.l_margin, w=page_w, h=80)
            self.ln(4)
            os.unlink(tmp_path)
        except Exception as e:
            self.set_text_color(255, 100, 100)
            self.set_font('Courier', '', 8)
            self.cell(0, 6, f'  [Image unavailable: {str(e)}]', new_x='LMARGIN', new_y='NEXT')


def generate_report(
    report_data: dict,
    output_path: str = None
) -> str:
    """
    Generate a PhantomEye PDF intelligence report.

    report_data keys:
    - session_id: str
    - total_persons: int
    - duration_seconds: int
    - loitering_alerts: int
    - detections: list of dicts {id, emotion, gender, age, dwell_seconds, loitering, weapon}
    - heatmap_img: np.ndarray or None
    - frame_sample: np.ndarray or None
    - weapon_detections: list of dicts {class_name, confidence}
    - nl_query: str or None
    - nl_result: str or None
    """

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("outputs", f"phantomeye_report_{ts}.pdf")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "outputs", exist_ok=True)

    pdf = PhantomEyeReport()
    pdf.set_margins(15, 25, 15)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Session Overview
    pdf.section_title("SESSION OVERVIEW")
    pdf.key_value("Session ID", report_data.get("session_id", "N/A"))
    pdf.key_value("Report Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.key_value("Total Persons Detected", str(report_data.get("total_persons", 0)))
    pdf.key_value("Session Duration", f"{report_data.get('duration_seconds', 0)} seconds")
    pdf.key_value("Loitering Alerts", str(report_data.get("loitering_alerts", 0)))
    pdf.key_value("Weapon Detections", str(len(report_data.get("weapon_detections", []))))
    pdf.ln(4)

    # Threat Summary
    weapon_dets = report_data.get("weapon_detections", [])
    if weapon_dets:
        pdf.section_title("THREAT ALERT -- WEAPONS DETECTED")
        pdf.set_text_color(255, 80, 80)
        pdf.set_font('Courier', 'B', 9)
        pdf.cell(0, 6, f'  !! {len(weapon_dets)} WEAPON(S) DETECTED -- IMMEDIATE ATTENTION REQUIRED', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(2)
        for w in weapon_dets:
            pdf.set_text_color(255, 150, 150)
            pdf.set_font('Courier', '', 9)
            pdf.cell(0, 6, f'  -> {w["class_name"]}  |  Confidence: {w["confidence"]:.0%}', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(4)

    # Person Intelligence Table
    detections = report_data.get("detections", [])
    if detections:
        pdf.section_title("SUBJECT INTELLIGENCE LOG")
        pdf.set_font('Courier', 'B', 8)
        pdf.set_text_color(0, 255, 136)
        pdf.set_fill_color(0, 50, 25)
        pdf.cell(20, 7, 'ID', fill=True, new_x='RIGHT', new_y='LAST')
        pdf.cell(35, 7, 'EMOTION', fill=True, new_x='RIGHT', new_y='LAST')
        pdf.cell(30, 7, 'GENDER', fill=True, new_x='RIGHT', new_y='LAST')
        pdf.cell(20, 7, 'AGE', fill=True, new_x='RIGHT', new_y='LAST')
        pdf.cell(35, 7, 'DWELL(s)', fill=True, new_x='RIGHT', new_y='LAST')
        pdf.cell(40, 7, 'LOITERING', fill=True, new_x='LMARGIN', new_y='NEXT')

        for i, d in enumerate(detections):
            fill = i % 2 == 0
            pdf.set_fill_color(0, 20, 10) if fill else pdf.set_fill_color(0, 30, 15)
            pdf.set_text_color(200, 255, 200)
            pdf.set_font('Courier', '', 8)
            pdf.cell(20, 6, str(d.get('id', '-')), fill=True, new_x='RIGHT', new_y='LAST')
            pdf.cell(35, 6, str(d.get('emotion', '-')).upper(), fill=True, new_x='RIGHT', new_y='LAST')
            pdf.cell(30, 6, str(d.get('gender', '-')), fill=True, new_x='RIGHT', new_y='LAST')
            pdf.cell(20, 6, str(d.get('age', '-')), fill=True, new_x='RIGHT', new_y='LAST')
            pdf.cell(35, 6, str(d.get('dwell_seconds', '-')), fill=True, new_x='RIGHT', new_y='LAST')
            loiter = 'YES !' if d.get('loitering') else 'NO'
            pdf.set_text_color(255, 100, 100) if d.get('loitering') else pdf.set_text_color(0, 255, 136)
            pdf.cell(40, 6, loiter, fill=True, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(4)

    # Heatmap
    if report_data.get("heatmap_img") is not None:
        pdf.add_image_section("BEHAVIORAL HEATMAP", report_data["heatmap_img"])

    # Frame Sample
    if report_data.get("frame_sample") is not None:
        pdf.add_image_section("SCENE SNAPSHOT", report_data["frame_sample"])

    # NL Query
    if report_data.get("nl_query"):
        pdf.section_title("NATURAL LANGUAGE QUERY LOG")
        pdf.key_value("Query", report_data.get("nl_query", ""))
        pdf.key_value("Result", report_data.get("nl_result", ""))
        pdf.ln(4)

    # Footer note
    pdf.set_text_color(0, 100, 50)
    pdf.set_font('Courier', 'I', 7)
    pdf.cell(0, 6, '  This report was automatically generated by PhantomEye AI Surveillance Intelligence System.', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 6, '  All data is session-based and was not stored server-side. For investigative use only.', new_x='LMARGIN', new_y='NEXT')

    pdf.output(output_path)
    return output_path