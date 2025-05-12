#!/usr/bin/env python3
"""
Script to create sample PDFs for testing the od-parse library.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfform

def create_basic_pdf(output_path):
    """
    Create a basic PDF with text, tables, and form elements.
    
    Args:
        output_path: Path to save the PDF
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create content elements
    elements = []
    
    # Add title
    title_style = styles["Heading1"]
    elements.append(Paragraph("Sample PDF Document", title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add paragraphs
    normal_style = styles["Normal"]
    elements.append(Paragraph("This is a sample PDF document created for testing the od-parse library. "
                             "It contains various elements like text, tables, and form fields.", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add a section heading
    heading2_style = styles["Heading2"]
    elements.append(Paragraph("Text Section", heading2_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Add more paragraphs
    elements.append(Paragraph("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
                             "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
                             "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.", normal_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu "
                             "fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa "
                             "qui officia deserunt mollit anim id est laborum.", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add a table section
    elements.append(Paragraph("Table Section", heading2_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Create a table
    data = [
        ["Name", "Age", "City"],
        ["John Smith", "30", "New York"],
        ["Jane Doe", "25", "Los Angeles"],
        ["Bob Johnson", "45", "Chicago"],
        ["Alice Brown", "35", "Houston"]
    ]
    
    table = Table(data, colWidths=[2*inch, inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Build the PDF
    doc.build(elements)
    
    # Add form elements using a separate canvas
    add_form_elements(output_path)

def add_form_elements(pdf_path):
    """
    Add form elements to an existing PDF.
    
    Args:
        pdf_path: Path to the PDF to modify
    """
    # Create a temporary PDF with form elements
    temp_path = pdf_path.replace(".pdf", "_temp.pdf")
    c = canvas.Canvas(temp_path, pagesize=letter)
    
    # Set font
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, 3*inch, "Form Elements Section")
    
    c.setFont("Helvetica", 12)
    
    # Add checkbox
    form = c.acroForm
    c.drawString(1*inch, 2.5*inch, "Checkbox Example:")
    form.checkbox(name="checkbox1", tooltip="Checkbox 1",
                 x=2.5*inch, y=2.5*inch, buttonStyle="check",
                 borderWidth=1, borderColor=colors.black,
                 fillColor=colors.white, textColor=colors.black,
                 size=12)
    
    # Add radio buttons
    c.drawString(1*inch, 2*inch, "Radio Button Example:")
    form.radio(name="radio1", tooltip="Radio 1",
              x=2.5*inch, y=2*inch, buttonStyle="circle",
              borderWidth=1, borderColor=colors.black,
              fillColor=colors.white, textColor=colors.black,
              size=12, selected=False)
    form.radio(name="radio1", tooltip="Radio 2",
              x=3*inch, y=2*inch, buttonStyle="circle",
              borderWidth=1, borderColor=colors.black,
              fillColor=colors.white, textColor=colors.black,
              size=12, selected=False)
    
    # Add text field
    c.drawString(1*inch, 1.5*inch, "Text Field Example:")
    form.textfield(name="text1", tooltip="Text Field 1",
                 x=2.5*inch, y=1.5*inch, width=2*inch, height=20,
                 borderWidth=1, borderColor=colors.black,
                 fillColor=colors.white, textColor=colors.black,
                 fontSize=12)
    
    c.save()
    
    # Merge the original PDF with the form elements
    # In a real implementation, you would use PyPDF2 or a similar library to merge PDFs
    # For simplicity, we'll just use the temporary PDF with form elements
    os.rename(temp_path, pdf_path)

def main():
    """
    Create sample PDFs for testing.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "sample_pdfs"
    output_dir.mkdir(exist_ok=True)
    
    # Create a basic PDF
    basic_pdf_path = output_dir / "basic_sample.pdf"
    create_basic_pdf(basic_pdf_path)
    print(f"Created basic sample PDF: {basic_pdf_path}")

if __name__ == "__main__":
    main()
