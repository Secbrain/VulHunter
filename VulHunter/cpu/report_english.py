# -*- coding:utf-8 -*-

import os
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.platypus.flowables import Macro
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.charts.legends import Legend
from collections import OrderedDict
from reportlab.lib.enums import TA_JUSTIFY

pdfmetrics.registerFont(TTFont('hei', "MSYH.TTC"))
pdfmetrics.registerFont(TTFont('roman', "simsun.ttc"))
pdfmetrics.registerFont(TTFont('heiti', "simhei.ttf"))
pdfmetrics.registerFont(TTFont('roman', "times.ttf"))

time_start = None
auditid = None

class NumberedCanvasEnglish(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._codes = []
    def showPage(self):
        self._codes.append({'code': self._code, 'stack': self._codeStack})
        self._startPage()
    def save(self):
        """add page info to each page (page x of y)"""
        # reset page counter 
        self._pageNumber = 0
        global time_start, auditid
        time_ym = time.strftime('%b', time_start)+' '+time.strftime('%Y', time_start)
        ...


class ReportEnglish():

	title_style = ParagraphStyle(name="TitleStyle", fontName="hei", fontSize=14, alignment=TA_LEFT,leading=20,spaceAfter=10,spaceBefore=10,textColor=colors.HexColor(0x256BA6),)
	sub_title_style = ParagraphStyle(name="SubTitleStyle", fontName="hei", fontSize=12,
	                                      textColor=colors.HexColor(0x256BA6), alignment=TA_LEFT, spaceAfter=7,spaceBefore=2,)
	sub_sub_title_style = ParagraphStyle(name="SubTitleStyle", fontName="hei", fontSize=12,
	                                      textColor=colors.black, alignment=TA_LEFT, spaceAfter=8,spaceBefore=5,)
	sub_title_style_romanbold = ParagraphStyle(name="SubTitleStyleRomanbold", fontName="hei", fontSize=12,
	                                      textColor=colors.HexColor(0x256BA6), alignment=TA_LEFT, spaceAfter=7,spaceBefore=2,)
	content_daoyin_style = ParagraphStyle(name="ContentDaoyinStyle", fontName="hei", fontSize=12, leading=20,
	                                    wordWrap = 'CJK', firstLineIndent = 24)
	content_daoyin_style_red = ParagraphStyle(name="ContentDaoyinStyleRed", fontName="hei", fontSize=12, leading=20,
	                                    wordWrap = 'CJK', textColor=colors.red)
	content_style = ParagraphStyle(name="ContentStyle", fontName="roman", fontSize=12, leading=20, alignment=TA_JUSTIFY,
	                                    wordWrap = 'CJK', firstLineIndent = 24)
	content_style_noindent = ParagraphStyle(name="ContentStyleNoindent", fontName="roman", fontSize=12, leading=20,
	                                    wordWrap = 'CJK')
	content_style_roman = ParagraphStyle(name="ContentStyleRoman", fontName="roman", fontSize=12, leading=20,
	                                    wordWrap = 'CJK', firstLineIndent = 24)
	content_style_codeadd = ParagraphStyle(name="ContentStyle", fontName="roman", fontSize=10.5, leading=20,
	                                    wordWrap = 'CJK', firstLineIndent = 24)
	content_style_red = ParagraphStyle(name="ContentStyleRed", fontName="roman", fontSize=12, leading=20,
	                                    wordWrap = 'CJK', firstLineIndent = 24, textColor=colors.red)
	foot_style = ParagraphStyle(name="FootStyle", fontName="hei", fontSize=10.5, textColor=colors.HexColor(0xB4B4B4),
	                                 leading=25, spaceAfter=20, alignment=TA_CENTER, )
	table_title_style = ParagraphStyle(name="TableTitleStyle", fontName="hei", fontSize=10.5, leading=20,
	                                        spaceAfter=2, alignment=TA_CENTER, )
	graph_title_style = ParagraphStyle(name="GraphTitleStyle", fontName="hei", fontSize=10.5, leading=20,
	                                        spaceBefore=7, alignment=TA_CENTER, )
	sub_table_style = ParagraphStyle(name="SubTableTitleStyle", fontName="hei", fontSize=10.5, leading=25,
	                                        spaceAfter=10, alignment=TA_LEFT, )
	code_style = ParagraphStyle(name="CodeStyle", fontName="hei", fontSize=9.5, leading=12,
	                                        spaceBefore=5, spaceAfter=5, alignment=TA_LEFT,borderWidth=0.3,borderColor = colors.HexColor(0x256BA6), wordWrap = 'CJK', )
	...

	# def __init__(self):
	# 	"""
		
	# 	"""
		

	def _output(self, result_maps, filename, time_start_para, auditcontent, report_path, contracts_names, auditid_para, bytecodes):
		global time_start, auditid
		time_start = time_start_para
		auditid = auditid_para

		story = []

		story.append(PageBreak())
		story.append(Paragraph("0x01 Summary Information", self.title_style))
		story.append(Paragraph("The VulHunter (VH, for short) platform received this smart contract security audit application and audited the contract in "+time.strftime('%b', time_start)+" "+time.strftime('%Y', time_start)+".", self.content_style))
		story.append(Paragraph('It is necessary to declare that VH only issues this report in respect of facts that have occurred or existed before the issuance of this report, and undertakes corresponding responsibilities for this. For the facts that occur or exist in the future, VH is unable to judge the security status of its smart contract, and will not be responsible for it. The security audit analysis and other content made in this report are based on the documents and information provided to smart analysis team by the information provider as of the issuance of this report (referred to as "provided information"). VH hypothesis: There is no missing, tampered, deleted or concealed information in the mentioned information. If the information that has been mentioned is missing, tampered with, deleted, concealed or reflected does not match the actual situation, VulHunter shall not be liable for any losses and adverse effects caused thereby.', self.content_style))
		story.append(Spacer(1, 1.5 * mm))
		story.append(Paragraph("Table 1 Contract audit information", self.table_title_style))

		...
	
	#output the main audit result
	def _output_main(self, result_maps, filename, time_start_para, auditcontent, report_path, contracts_names, auditid_para, bytecodes):
		global time_start, auditid
		time_start = time_start_para
		auditid = auditid_para

		story = []
		...
		doc = SimpleDocTemplate(report_path,
		                        pagesize=A4,
		                        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=27 * mm, bottomMargin=25 * mm)
		doc.build(story,canvasmaker=NumberedCanvasEnglish)
		print("The audit report has been saved to "+report_path+".")