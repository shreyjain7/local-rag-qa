#!/usr/bin/env python
"""
Convert PROJECT_DOCUMENTATION.md to PDF
Generates a professional PDF document
"""

import sys
from pathlib import Path

def install_reportlab():
    """Install reportlab if not already installed"""
    try:
        import reportlab
        return True
    except ImportError:
        print("Installing reportlab for PDF generation...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
            return True
        except:
            print("Could not install reportlab. Trying alternative...")
            return False

def generate_pdf():
    """Generate PDF from markdown"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from datetime import datetime
    except ImportError:
        print("Error: reportlab not available")
        return False
    
    # Read markdown file
    md_file = Path("PROJECT_DOCUMENTATION.md")
    if not md_file.exists():
        print(f"Error: {md_file} not found")
        return False
    
    print(f"Reading {md_file}...")
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF
    pdf_file = "PROJECT_DOCUMENTATION.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                          rightMargin=0.75*inch, leftMargin=0.75*inch,
                          topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#2e5c8a'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Courier',
        textColor=colors.HexColor('#333333'),
        backColor=colors.HexColor('#f5f5f5'),
        leftIndent=12,
        spaceAfter=6
    )
    
    # Build story
    story = []
    
    # Title page
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("Local RAG-Based Document Q&A System", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Project Documentation & Technical Guide", 
                          ParagraphStyle('Subtitle', parent=styles['Normal'], 
                                       fontSize=14, alignment=TA_CENTER, 
                                       textColor=colors.HexColor('#666666'))))
    story.append(Spacer(1, 1*inch))
    
    # Add metadata
    story.append(Paragraph("Created: June 30, 2026", normal_style))
    story.append(Paragraph("Status: Fully Functional", normal_style))
    story.append(Paragraph("Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), normal_style))
    
    story.append(PageBreak())
    
    # Parse markdown and add to story
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Skip empty lines
        if not line:
            story.append(Spacer(1, 0.1*inch))
            i += 1
            continue
        
        # Headers
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            story.append(Paragraph(text, heading1_style))
            i += 1
            continue
        
        if line.startswith('## '):
            text = line[3:].strip()
            story.append(Paragraph(text, heading2_style))
            i += 1
            continue
        
        if line.startswith('### '):
            text = line[4:].strip()
            story.append(Paragraph(text, 
                                  ParagraphStyle('Heading3', parent=styles['Normal'],
                                               fontSize=11, fontName='Helvetica-Bold',
                                               textColor=colors.HexColor('#3d6b99'))))
            i += 1
            continue
        
        # Code blocks
        if line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            
            code_text = '\n'.join(code_lines).strip()
            if code_text:
                for code_line in code_text.split('\n'):
                    safe_text = code_line.replace('<', '[').replace('>', ']') if code_line else " "
                    story.append(Paragraph(safe_text, code_style))
                story.append(Spacer(1, 0.1*inch))
            
            i += 1
            continue
        
        # Bullet points
        if line.startswith('- '):
            text = line[2:].strip()
            text = text.replace('<', '[').replace('>', ']')
            story.append(Paragraph("• " + text, normal_style))
            i += 1
            continue
        
        # Regular paragraph - escape special characters
        safe_line = line.replace('<', '[').replace('>', ']')
        if safe_line.strip():
            story.append(Paragraph(safe_line, normal_style))
        
        i += 1
    
    # Build PDF
    print(f"Generating {pdf_file}...")
    try:
        doc.build(story)
        print(f"✓ PDF created successfully: {pdf_file}")
        print(f"  Location: {Path(pdf_file).absolute()}")
        return True
    except Exception as e:
        print(f"Error creating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("="*60)
    print("PROJECT DOCUMENTATION - PDF Generator")
    print("="*60)
    
    # Try to install reportlab if needed
    if not install_reportlab():
        print("\n⚠ PDF generation unavailable")
        print("Install manually with: pip install reportlab")
        print("\nMarkdown documentation available at: PROJECT_DOCUMENTATION.md")
        return False
    
    # Generate PDF
    if generate_pdf():
        print("\n✅ PDF generation complete!")
        return True
    else:
        print("\n❌ PDF generation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
