# -*- coding:utf-8 -*-

import os
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.platypus.flowables import Macro
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Rect, Image
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.charts.legends import Legend
from collections import OrderedDict
from reportlab.lib.enums import TA_JUSTIFY

import reportlab.lib, reportlab.platypus

pdfmetrics.registerFont(TTFont('hei', "MSYH.TTC"))
pdfmetrics.registerFont(TTFont('roman', "simsun.ttc"))
pdfmetrics.registerFont(TTFont('heiti', "simhei.ttf"))
pdfmetrics.registerFont(TTFont('roman', "times.ttf"))
month_convert = {'1月':'January','2月':'February','3月':'March','4月':'April','5月':'May','6月':'June','7月':'July','8月':'August','9月':'September','10月':'October','11月':'November','12月':'December'}

time_start = None
auditid = None

class flowable_fig(reportlab.platypus.Flowable):
    def __init__(self, imgdata):
        reportlab.platypus.Flowable.__init__(self)
        self.img = reportlab.lib.utils.ImageReader(imgdata)

    def draw(self):
        self.canv.drawImage(self.img, 0, 0, height = -2*inch, width=4*inch)

class NumberedCanvasEnglish(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._codes = []
        # self._saved_page_states = []
    def showPage(self):
        self._codes.append({'code': self._code, 'stack': self._codeStack})
        # self._saved_page_states = []
        self._startPage()
    def save(self):
        """add page info to each page (page x of y)"""
        # reset page counter 
        self._pageNumber = 0
        global time_start, auditid
        if '月' in time.strftime('%b', time_start):
            time_ym = month_convert[time.strftime('%b', time_start)]+' '+time.strftime('%Y', time_start)
        else:
            time_ym = time.strftime('%b', time_start)+' '+time.strftime('%Y', time_start)
        # num_pages = len(self._saved_page_states)
        # print(len(self._saved_page_states))
        # for state in self._saved_page_states:
        # print(len(self._codes))
        for code in self._codes:
        #     # recall saved page
            self._code = code['code']
            self._codeStack = code['stack']
            if self._pageNumber == 0:
                self.drawImage(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_english/report-positive.jpg'),0,0,A4[0],A4[1])
                self.setFillColorRGB(1,1,1) #choose your font colour
                self.setFont("hei", 20) #choose your font type and font size
                self.drawString(177, 396, 'Num: '+ auditid)
                self.drawString(177, 346, 'Date: '+ time.strftime("%Y-%m-%d", time_start))
            elif self._pageNumber == len(self._codes)-1:
                self.drawImage(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_english/report-back.jpg'),0,0,A4[0],A4[1])
            else:
                self.setFillColorRGB(0.15,0.42,0.65)#37,107,166
                self.setStrokeColorRGB(0.15,0.42,0.65)
                self.rect(65, 775, 20, 50, stroke=1, fill=1)
                self.setFont("hei", 12) #choose your font type and font size
                self.drawString(90, 802, 'Smart Contract Audit Report')
                self.drawString(90, 780, time_ym)
                self.drawImage(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_english/logo.jpg'), 275, 780, width=100,height=40)
                self.drawImage(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_english/yemei.jpg'), 385, 785, width=140,height=30)
                self.line(180,775,530,775)
                self.setFont("hei", 10.5)
                self.drawCentredString(295, 30,
                    "%(this)i / %(end)i" % {
                       'this': self._pageNumber,
                       'end': len(self._codes)-2,
                    }
                )
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)


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
	# figure_style = ParagraphStyle(name="FigureStyle", alignment=TA_CENTER, )
	basic_style = TableStyle([('FONTNAME', (0, 0), (-1, -1), 'hei'),
	                               ('FONTSIZE', (0, 0), (-1, -1), 12),
	                               ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
	                               ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
	                               ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
	                               # 'SPAN' (列,行)坐标
	                               ('SPAN', (1, 0), (3, 0)),
	                               ('SPAN', (1, 1), (3, 1)),
	                               ('SPAN', (1, 2), (3, 2)),
	                               ('SPAN', (1, 5), (3, 5)),
	                               ('SPAN', (1, 6), (3, 6)),
	                               ('SPAN', (1, 7), (3, 7)),
	                               ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
	                               ])
	common_style = TableStyle([('FONTNAME', (0, 0), (-1, 0), 'hei'),
	                           ('FONTNAME', (1, 1), (-1, -1), 'roman'),
	                           ('FONTNAME', (0, 1), (0, -1), 'hei'),
	                              ('FONTSIZE', (0, 0), (-1, -1), 12),
	                              ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
	                              ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
	                              ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
	                           ('LINEBEFORE', (0, 0), (0, -1), 0.1, colors.grey),  # 设置表格左边线颜色为灰色，线宽为0.1
	                              ('GRID', (0, 0), (-1, -1), 0.1, colors.grey),
	                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # 设置表格内文字颜色
	                           ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
	                           ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472c4')),  # 设置第一行背景颜色
	                            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#d9e2f3')),  # 设置第二行背景颜色
	                           ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#d9e2f3')),
	                           ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#d9e2f3')),
	                           ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#d9e2f3')),
	                             ])
	common_style_1 = TableStyle([('FONTNAME', (0, 0), (-1, 0), 'hei'),
	                           ('FONTNAME', (0, 1), (0, -1), 'hei'),
	                           ('FONTNAME', (1, 1), (-1, -1), 'roman'),
	                              ('FONTSIZE', (0, 0), (-1, -1), 12),
	                              ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
	                              ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
	                              ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
	                           ('LINEBEFORE', (0, 0), (0, -1), 0.1, colors.grey),  # 设置表格左边线颜色为灰色，线宽为0.1
	                              ('GRID', (0, 0), (-1, -1), 0.1, colors.grey),
	                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # 设置表格内文字颜色
	                           ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor('#E61A1A')),
	                             ('TEXTCOLOR', (1, 1), (1, -1), colors.HexColor('#FF6600')),
	                             ('TEXTCOLOR', (2, 1), (2, -1), colors.HexColor('#DDB822')),
	                             ('TEXTCOLOR', (3, 1), (3, -1), colors.HexColor('#ff66ff')),
	                             ('TEXTCOLOR', (4, 1), (4, -1), colors.HexColor('#22DDDD')),
	#                              ('TEXTCOLOR', (5, 1), (5, -1), colors.HexColor('#2BD591')),
	                           ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472c4')),  # 设置第一行背景颜色
	                            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#d9e2f3')),  # 设置第二行背景颜色
	                             ('SPAN', (0, 0), (-1, 0)),
	                             ])
	common_style_result_all_type = [
	                            ('FONTNAME', (0, 0), (-1, 0), 'hei'),
	                            ('FONTNAME', (0, 1), (0, -1), 'roman'),
	                           ('FONTNAME', (1, 1), (1, -1), 'hei'),
	                           ('FONTNAME', (2, 1), (4, -1), 'roman'),
	                            ('FONTNAME', (5, 1), (5, -1), 'hei'),
	                              ('FONTSIZE', (0, 0), (-1, -1), 9),
	                              # ('FONTSIZE', (2, 0), (2, -1), 7.5),
	                              ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
	                              ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
	                              ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
	                           ('LINEBEFORE', (0, 0), (0, -1), 0.1, colors.grey),  # 设置表格左边线颜色为灰色，线宽为0.1
	                              ('GRID', (0, 0), (-1, -1), 0.1, colors.grey),
	                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # 设置表格内文字颜色
	                           ('TEXTCOLOR', (0, 1), (2, -1), colors.black),
	                            ('TEXTCOLOR', (4, 1), (4, -1), colors.black),
	                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472c4'))  # 设置第一行背景颜色
	                             ]
	content_colors = {'高危':'#E61A1A','中危':'#FF6600','低危':'#DDB822','提醒':'#ff66ff','优化':'#22DDDD','通过':'#2BD591'}
	table_result = [['ID', 'Pattern', 'Description', 'Severity', 'Confidence', 'Status/Num', 'Description', 'Scenarios', 'Scenarios_supplement', 'Recommendation', 'Severity'], [1, 'reentrancy-eth', 'Re-entry vulnerabilities (Ethereum theft)', 'High', 'probably', 'Pass', 'A reentrancy error was detected. This is the reentry of ether. Through re-entry, the account balance can be maliciously withdrawn, resulting in losses. Do not report re-reporting that does not involve Ether (please refer to "reentrancy-no-eth")', 'function withdrawBalance(){<br/>&#160;&#160;&#160;&#160;// send userBalance[msg.sender] Ether to msg.sender<br/>&#160;&#160;&#160;&#160;// if mgs.sender is a contract, it will call its fallback function<br/>&#160;&#160;&#160;&#160;if( ! (msg.sender.call.value(userBalance[msg.sender])() ) ){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;throw;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;userBalance[msg.sender] = 0;<br/>}', 'Bob used the reentrance vulnerability to call `withdrawBalance` multiple times and withdrew more than he originally deposited into the contract.', 'Can adopt check-effects-interactions mode.', 'High'], [2, 'controlled-array-length', 'Length is allocated directly', 'High', 'probably', 'Pass', 'Detect direct allocation of array length.', 'contract A {<br/>&#160;&#160;&#160;&#160;uint[] testArray; // dynamic size array<br/>&#160;&#160;&#160;&#160;function f(uint usersCount) public {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;testArray.length = usersCount;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function g(uint userIndex, uint val) public {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;testArray[userIndex] = val;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Contract storage/state variables are indexed by 256-bit integers. Users can set the array length to 2 ** 256-1 to index all storage slots. In the above example, you can call function f to set the length of the array, and then call function g to control any storage slots needed. Please note that the storage slots here are indexed by the hash of the indexer. Nonetheless, all storage will still be accessible and can be controlled by an attacker.', 'It is not allowed to set the length of the array directly; instead, choose to add values as needed. Otherwise, please check the contract thoroughly to ensure that the user-controlled variables cannot reach the array length allocation.', 'High'], [3, 'suicidal', 'Check if anyone can break the contract', 'High', 'exactly', 'Pass', 'Due to lack of access control or insufficient access control, malicious parties can self-destruct the contract. Calling selfdestruct/suicide lacks protection.', 'contract Suicidal{<br/>&#160;&#160;&#160;&#160;function kill() public{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;selfdestruct(msg.sender);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Bob calls the "kill" function and breaks the contract.', 'Protect access to all sensitive functions.', 'High'], [4, 'controlled-delegatecall', 'The delegate address out of control', 'High', 'probably', 'Pass', 'Delegate the call or call code to an address controlled by the user. The address of Delegatecall is not necessarily trusted, it is still a problem of access control, and the address is not checked.', 'contract Delegatecall{<br/>&#160;&#160;&#160;&#160;function delegate(address to, bytes data){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;to.delegatecall(data);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Bob calls `delegate` and delegates the execution of the malicious contract to him. As a result, Bob withdraws the funds from the contract and destroys the contract.', 'Avoid using `delegatecall`. If you use it, please only target trusted destinations.', 'High'], [5, 'arbitrary-send', 'Check if Ether can be sent to any address', 'High', 'probably', 'Pass', 'The call to the function that sends Ether to an arbitrary address has not been reviewed.', 'contract ArbitrarySend{<br/>&#160;&#160;&#160;&#160;address destination;<br/>&#160;&#160;&#160;&#160;function setDestination(){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destination = msg.sender;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function withdraw() public{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destination.transfer(this.balance);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Bob calls setDestination and withdraw, and as a result, he withdraws the balance of the contract.', 'Ensure that no user can withdraw unauthorized funds.', 'High'], [6, 'tod', 'Transaction sequence dependence for receivers/ethers', 'High', 'probably', 'Pass', "Mainly about the receiver's exception. A person who is running an Ethereum node can tell which transactions are going to occur before they are finalized.A race condition vulnerability occurs when code depends on the order of the transactions submitted to it.", 'pragma solidity ^0.4.5;<br/>contract StandardToken is ERC20, BasicToken {<br/>&#160;&#160;&#160;&#160;...<br/>&#160;&#160;&#160;&#160;function approve(address _spender, uint256 _value) public returns (bool) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;allowed[msg.sender][_spender] = _value;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;Approval(msg.sender, _spender, _value);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return true;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;...<br/>}', '', 'Before performing the approve change, reset the value to zero first, and then perform the change operation.', 'High'], [7, 'uninitialized-state', 'Check for uninitialized state variables', 'High', 'exactly', 'Pass', 'Uninitialized state variables can lead to intentional or unintentional vulnerabilities.', 'contract Uninitialized{<br/>&#160;&#160;&#160;&#160;address destination;<br/>&#160;&#160;&#160;&#160;function transfer() payable public{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destination.transfer(msg.value);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Bob calls "transfer". As a result, the ether is sent to the address "0x0" and is lost.', 'Initialize all variables. If you want to initialize a variable to zero, set it explicitly to zero.', 'High'], [8, 'parity-multisig-bug', 'Check for multi-signature vulnerabilities', 'High', 'probably', 'Pass', 'Multi-signature vulnerability. Hackers can use the initWallet function to call the initMultiowned function to obtain the identity of the contract owner.', 'contract WalletLibrary_bad is WalletEvents {<br/>&#160;&#160;&#160;&#160;function initWallet(address[] _owners, uint _required, uint _daylimit) { <br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;initDaylimit(_daylimit); <br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;initMultiowned(_owners, _required);<br/>&#160;&#160;&#160;&#160;}  // kills the contract sending everything to `_to`.<br/>&#160;&#160;&#160;&#160;function initMultiowned(address[] _owners, uint _required) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_numOwners = _owners.length + 1;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_owners[1] = uint(msg.sender);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_ownerIndex[uint(msg.sender)] = 1; <br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;for (uint i = 0; i < _owners.length; ++i)<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_owners[2 + i] = uint(_owners[i]);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_ownerIndex[uint(_owners[i])] = 2 + i;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;m_required = _required;<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'InitWallet, initDaylimit and initMultiowned add internal limited types to prohibit external calls: or if only_uninitMultiowned (m_numOwners) is detected in initMultiowned, no error will occur. The number of initializations can also be reviewed by judging m_numOwners.', 'High'], [9, 'incorrect-equality', 'Check the strict equality of danger', 'Medium', 'exactly', 'Pass', 'Using strict equality (== and !=), an attacker can easily manipulate these equality. Specifically: the opponent can forcefully send Ether to any address through selfdestruct() or through mining, thereby invalidating the strict judgment.', 'contract Crowdsale{<br/>&#160;&#160;&#160;&#160;function fund_reached() public returns(bool){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return this.balance == 100 ether;<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Crowdsale relies on fund_reached to know when to stop the sale of tokens. Bob sends 0.1 ether. As a result, fund_reached is always false, and crowdsale is always true.', 'Do not use strict equality to determine whether an account has enough ether or tokens.', 'Medium'], [10, 'integer-overflow', 'Check for integer overflow', 'Medium', 'probably', 'Pass', 'When an arithmetic operation reaches the maximum or minimum size of the type, overflow/underflow will occur. For example, if a number is stored in the uint8 type, it means that the number is stored as an 8-bit unsigned number, ranging from 0 to 2^8-1. In computer programming, when an arithmetic operation attempts to create a value, an integer overflow occurs, and the value can be represented by a given number of bits-greater than the maximum value or less than the minimum value.', 'contract Intergeroverflow{<br/>&#160;&#160;&#160;&#160;function bad() {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;uint a;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;uint b;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;uint c = a + b;<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Use Safemath to perform integer arithmetic or verify calculated values.', 'Medium'], [11, 'unchecked-lowlevel', 'Check for uncensored low-level calls', 'Medium', 'probably', 'Pass', 'The low-level call to the external contract failed, and the return value was not judged. When sending ether at the same time, please check the return value and handle the error.', 'contract MyConc{<br/>&#160;&#160;&#160;&#160;function my_func(address payable dst) public payable{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;dst.call.value(msg.value)("");<br/>&#160;&#160;&#160;&#160;}<br/>}', 'The return value of the low-level call is not checked, so if the call fails, the ether will be locked in the contract. If you use low-level calls to block block operations, consider logging the failed calls.', 'Make sure to check or record the return value of low-level calls.', 'Medium'], [12, 'tx-origin', 'Check the dangerous use of tx.origin', 'Medium', 'probably', 'Pass', 'If a legitimate user interacts with a malicious contract, the protection based on tx.origin will be abused by the malicious contract.', 'contract TxOrigin {<br/>&#160;&#160;&#160;&#160;address owner = msg.sender;<br/>&#160;&#160;&#160;&#160;function bug() {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(tx.origin == owner);<br/>&#160;&#160;&#160;&#160;}<br/>}', "Bob is the owner of TxOrigin. Bob calls Eve's contract. Eve's contract is called TxOrigin and bypasses the protection of tx.origin.", 'Do not use `tx.origin` for authorization.', 'Medium'], [13, 'locked-ether', 'Whether the contract ether is locked', 'Medium', 'exactly', 'Pass', 'A contract programmed to receive ether (with the payable logo) should implement the method of withdrawing ether, that is, call transfer (recommended), send or call.value at least once.', 'pragma solidity 0.4.24;<br/>contract Locked{<br/>&#160;&#160;&#160;&#160;function receive() payable public{}<br/>}', 'All Ether sent to "Locked" will be lost.', 'Delete payable attributes or add withdrawal functions.', 'Medium'], [14, 'unchecked-send', 'Check unreviewed send', 'Medium', 'probably', 'Pass', 'Similar to unchecked-lowlevel, it is explained here that the return value of send and Highlevelcall is not checked.', 'contract MyConc{<br/>&#160;&#160;&#160;&#160;function my_func(address payable dst) public payable{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;dst.send(msg.value);<br/>&#160;&#160;&#160;&#160;}<br/>}', 'The return value of send is not checked, so if the send fails, the ether will be locked in the contract. If you use send to prevent block operations, please consider logging failed send.', 'Make sure to check or record the return value of send.', 'Medium'], [15, 'boolean-cst', 'Check for misuse of Boolean constants', 'Medium', 'probably', 'Pass', 'Detect abuse of Boolean constants. Bool variable is used incorrectly, here is the operation of bool variable.', 'contract A {<br/>&#160;&#160;&#160;&#160;function f(uint x) public {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if (false) { // bad!<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function g(bool b) public returns (bool) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return (b || true); // bad!<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>}', 'The Boolean constants in the code have very few legal uses. Other uses (as conditions in complex expressions) indicate the persistence of errors or error codes.', 'Verify and simplify the conditions.', 'Medium'], [16, 'erc721-interface', 'Check the wrong ERC721 interface', 'Medium', 'exactly', 'Pass', 'The return value of the "ERC721" function is incorrect. Interacting with these functions, the contract of solidity version> 0.4.22 will not be executed because of the lack of return value.', 'contract Token{<br/>&#160;&#160;&#160;&#160;function ownerOf(uint256 _tokenId) external view returns (bool);<br/>&#160;&#160;&#160;&#160;//...<br/>}', "Token.ownerOf does not return the expected boolean value. Bob deploys the token. Alice creates a contract to interact with, but uses the correct `ERC721` interface implementation. Alice's contract cannot interact with Bob's contract.", 'Set appropriate return values and vtypes for the defined ʻERC721` function.', 'Medium'], [17, 'erc20-interface', 'Check for wrong ERC20 interface', 'Medium', 'exactly', 'Pass', 'The return value of the "ERC20" function is incorrect. Interacting with these functions, the contract of solidity version> 0.4.22 will not be executed because of the lack of return value.', 'contract Token{<br/>&#160;&#160;&#160;&#160;function transfer(address to, uint value) external;<br/>&#160;&#160;&#160;&#160;//...<br/>}', "Token.transfer does not return the expected boolean value. Bob deploys the token. Alice creates a contract to interact with, but uses the correct `ERC20` interface implementation. Alice's contract cannot interact with Bob's contract.", 'Set appropriate return values and vtypes for the defined ʻERC20` function.', 'Medium'], [18, 'costly-loop', 'Check for too expensive loops', 'Low', 'possibly', 'Pass', 'Ethereum is a very resource-constrained environment. The price of each calculation step is several orders of magnitude higher than the price of the centralized provider. In addition, Ethereum miners impose limits on the total amount of natural gas consumed in the block. If array.length is large enough, the function exceeds the gas limit, and the transaction that calls the function will never be confirmed. If external participants influence array.length, this will become a security issue.', 'pragma solidity 0.4.24;<br/>contract PriceOracle {<br/>&#160;&#160;&#160;&#160;address internal owner;<br/>&#160;&#160;&#160;&#160;address[] public subscribers;<br/>&#160;&#160;&#160;&#160;mapping(address => uint) balances;<br/>&#160;&#160;&#160;&#160;uint internal constant PRICE = 10**15;<br/>&#160;&#160;&#160;&#160;function subscribe() payable external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;subscribers.push(msg.sender);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;balances[msg.sender] += msg.value;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function setPrice(uint price) external {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(msg.sender == owner);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;bytes memory data = abi.encodeWithSelector(SIGNATURE, price);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;for (uint i = 0; i < subscribers.length; i++) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if(balances[subscribers[i]] >= PRICE) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;balances[subscribers[i]] -= PRICE;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;subscribers[i].call.gas(50000)(data);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Please check the dynamic array of the loop carefully. If you find it can be exploited by an attacker, please change it to prevent the contract from executing too many loops and causing gas overflow and rollback.', 'Low'], [19, 'timestamp', 'The dangerous use of block.timestamp', 'Low', 'probably', 'Pass', 'There is a strict comparison with block.timestamp or now in the contract, and miners can benefit from block.timestamp.', 'contract Timestamp{<br/>&#160;&#160;&#160;&#160;event Time(uint);<br/>&#160;&#160;&#160;&#160;modifier onlyOwner {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.timestamp == 0);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;_;  <br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function bad0() external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.timestamp == 0);<br/>&#160;&#160;&#160;&#160;}<br/>}', "Bob's contract relies on the randomness of block.timestamp. Eve is a miner who manipulates block.timestamp to take advantage of Bob's contract.", 'Avoid relying on block.timestamp.', 'Low'], [20, 'block-other-parameters', 'Hazardous use variables (block.number etc.)', 'Low', 'probably', 'Pass', 'Contracts usually require access to time values \u200b\u200bto perform certain types of functions. block.number can let you know the current time or time increment, but in most cases it is not safe to use them. block.number The block time of Ethereum is usually about 14 seconds, so the time increment between blocks can be predicted. However, the lockout time is not fixed and may change due to various reasons (for example, fork reorganization and difficulty coefficient). Since the block time is variable, block.number should not rely on accurate time calculations. The ability to generate random numbers is very useful in various applications. An obvious example is a gambling DApp, where a pseudo-random number generator is used to select the winner. However, creating a sufficiently powerful source of randomness in Ethereum is very challenging. Using blockhash, block.difficulty and other areas is also unsafe because they are controlled by miners. If the stakes are high, the miner can mine a large number of blocks by renting hardware in a short period of time, select the block that needs to obtain the block hash value to win, and then discard all other blocks.', 'contract Otherparameters{<br/>&#160;&#160;&#160;&#160;event Number(uint);<br/>&#160;&#160;&#160;&#160;event Coinbase(address);<br/>&#160;&#160;&#160;&#160;event Difficulty(uint);<br/>&#160;&#160;&#160;&#160;event Gaslimit(uint);<br/>&#160;&#160;&#160;&#160;function bad0() external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.number == 20);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.coinbase == msg.sender);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.difficulty == 20);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;require(block.gaslimit == 20);<br/>&#160;&#160;&#160;&#160;}<br/>}', "The randomness of Bob's contract depends on block.number and so on. Eve is a miner who manipulates block.number and so on to use Bob's contract.", 'Avoid relying on block.number and other data that can be manipulated by miners.', 'Low'], [21, 'calls-loop', 'Check the external call in the loop', 'Low', 'probably', 'Pass', 'Check that the key access control ETH is transmitted cyclically. If at least one address cannot receive ETH (for example, it is a contract with a default fallback function), the entire transaction will be restored. Loss of parameters.', 'contract CallsInLoop{<br/>&#160;&#160;&#160;&#160;address[] destinations;<br/>&#160;&#160;&#160;&#160;constructor(address[] newDestinations) public{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destinations = newDestinations;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function bad() external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;for (uint i=0; i < destinations.length; i++){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;destinations[i].transfer(i);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;}<br/>}', 'If one of the destination addresses is restored by the rollback function, bad() will restore all, so all the work is wasted.', 'Try to avoid calling external contracts in the loop, and you can use the pull over push strategy.', 'Low'], [22, 'low-level-calls', 'Check low-level calls', 'Info', 'exactly', 'Pass', 'Label low-level methods such as call, delegatecall, and callcode, because these methods are easily exploited by attackers.', 'contract Sender {<br/>&#160;&#160;&#160;&#160;address owner;<br/>&#160;&#160;&#160;&#160;modifier onlyceshi() {<br/>&#160;&#160;&#160;&#160;owner.callcode(bytes4(keccak256("inc()")));<br/>&#160;&#160;&#160;&#160;_;<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function send(address _receiver) payable external {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;_receiver.call.value(msg.value).gas(7777)("");<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function sendceshi(address _receiver) payable external {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if(_receiver.call.value(msg.value).gas(7777)("")){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;revert();<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Avoid low-level calls. Check whether the call is successful. If the call is to sign a contract, please check whether the code exists.', 'Informational'], [23, 'erc20-indexed', 'ERC20 event parameter is missing indexed', 'Info', 'exactly', 'Pass', 'The address parameters of the "Transfer" and "Approval" events of the ERC-20 token standard shall include indexed.', 'contract ERC20Bad {<br/>&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;event Transfer(address from, address to, uint value);<br/>&#160;&#160;&#160;&#160;event Approval(address owner, address spender, uint value);<br/>&#160;&#160;&#160;&#160;// ...<br/>}', 'According to the definition of the ERC20 specification, the first two parameters of the Transfer and Approval events should carry the indexed keyword. If these keywords are not included, the parameter data will be excluded from the bloom filter of the transaction/block. Therefore, external tools searching for these parameters may ignore them and fail to index the logs in this token contract.', 'According to the ERC20 specification, the indexed keyword is added to the event parameter of the corresponding keyword.', 'Informational'], [24, 'erc20-throw', 'ERC20 throws an exception', 'Info', 'exactly', 'Pass', 'The function of the ERC-20 token standard should be thrown in the following special circumstances: if there are not enough tokens in the _from account balance to spend, it should be thrown; unless the _from account deliberately authorizes the sending of messages through some mechanism Otherwise, transferFrom should be thrown.', 'contract SomeToken {<br/>&#160;&#160;&#160;&#160;mapping(address => uint256) balances;<br/>&#160;&#160;&#160;&#160;event Transfer(address indexed _from, address indexed _to, uint256 _value);<br/>&#160;&#160;&#160;&#160;function transfer(address _to, uint _value) public returns (bool) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if (_value > balances[msg.sender] || _value > balances[_to] + _value) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return false;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;balances[msg.sender] = balances[msg.sender] - _value;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;balances[_to] = balances[_to] + _value;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;emit Transfer(msg.sender, _to, _value);<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return true;<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Add the corresponding throw method to ERC-20 tokens.', 'Informational'], [25, 'hardcoded', 'Check the legitimacy of the address', 'Info', 'probably', 'Pass', 'The contract contains an unknown address, which may be used for some malicious activities. Need to check the hard-coded address and its purpose. The address length is prone to errors, and the length of the address is not enough, it will not report an error, so it is very dangerous to write a mistake. Here is an identification.', 'contract C {<br/>&#160;&#160;&#160;&#160;function f(uint a, uint b) pure returns (address) {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;address public multisig = 0xf64B584972FE6055a770477670208d737Fff282f;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;return multisig;<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Check carefully whether the address is wrong, and if there is an error, please take the time to correct it.', 'Informational'], [26, 'array-instead-bytes', 'The byte array can be replaced with bytes', 'Opt', 'exactly', 'Pass', 'byte[] can be converted to bytes to save gas resources.', 'pragma solidity 0.4.24;<br/>contract C {<br/>&#160;&#160;&#160;&#160;byte[] someVariable;<br/>&#160;&#160;&#160;&#160;...<br/>}', '', 'Replacing byte[] with bytes can save gas.', 'Optimization'], [27, 'unused-state', 'Check unused state variables', 'Opt', 'exactly', 'Pass', 'Unused variables are allowed in Solidity, and they do not pose direct security issues. The best practice is to avoid them as much as possible: resulting in increased calculations (and unnecessary gas consumption) means errors or incorrect data structures, and usually means poor code quality leads to code noise and reduces code readability.', 'contract A{<br/>&#160;&#160;&#160;&#160;address unused;<br/>&#160;&#160;&#160;&#160;address public unused2;<br/>&#160;&#160;&#160;&#160;address private unused3;<br/>&#160;&#160;&#160;&#160;address unused4;<br/>&#160;&#160;&#160;&#160;address used;<br/>&#160;&#160;&#160;&#160;function ceshi1 () external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;unused3 = address(0);<br/>&#160;&#160;&#160;&#160;}<br/>}', '', 'Delete unused state variables.', 'Optimization'], [28, 'costly-operations-loop', 'Expensive operations in the loop', 'Opt', 'probably', 'Pass', 'Expensive operations within the loop.', 'contract CostlyOperationsInLoop{<br/>&#160;&#160;&#160;&#160;uint loop_count = 100;<br/>&#160;&#160;&#160;&#160;uint state_variable=0;<br/>&#160;&#160;&#160;&#160;function bad() external{<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;for (uint i=0; i < loop_count; i++){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;state_variable++;<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;function good() external{<br/>&#160;&#160;&#160;&#160;  uint local_variable = state_variable;<br/>&#160;&#160;&#160;&#160;  for (uint i=0; i < loop_count; i++){<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;local_variable++;<br/>&#160;&#160;&#160;&#160;  }<br/>&#160;&#160;&#160;&#160;  state_variable = local_variable;<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Due to the expensive SSTOREs, the incremental state variables in the loop will generate a large amount of gas, which may result in insufficient gas.', 'This test is a state variable in the loop. The state variable costs more gas than the local variable. This was not tested before, and only length was tested before.', 'Optimization'], [29, 'send-transfer', 'Check Transfe to replace Send', 'Opt', 'exactly', 'Pass', 'The recommended way to perform the check of Ether payment is addr.transfer(x). If the transfer fails, an exception is automatically raised.', 'if(!addr.send(42 ether)) {<br/>&#160;&#160;&#160;&#160;revert();<br/>}', '', 'It is safer to use transfer instead of send.', 'Optimization'], [30, 'boolean-equal', 'Check comparison with boolean constant', 'Opt', 'exactly', 'Pass', "Check the comparison of Boolean constants. There is no need to compare with true and false, so it's superfluous (gas consumption).", 'contract A {<br/>&#160;&#160;&#160;&#160;function f(bool x) public {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;if (x == true) { // bad!<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;   // ...<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;}<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;// ...<br/>&#160;&#160;&#160;&#160;}<br/>}', 'Boolean constants can be used directly without comparison with true or false.', 'Delete the equation equal to the Boolean constant.', 'Optimization'], [31, 'external-function', 'Public functions can be declared as external', 'Opt', 'exactly', 'Pass', 'Functions with public visibility modifiers are not called internally. Changing the visibility level to an external level can improve the readability of the code. In addition, in many cases, functions that use external visibility modifiers cost less gas than functions that use public visibility modifiers.', 'contract ContractWithFunctionCalledSuper {<br/>&#160;&#160;&#160;&#160;function callWithSuper() {<br/>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;uint256 i = 0;<br/>&#160;&#160;&#160;&#160;}<br/>}', 'The callWithSuper() function can be declared as external visibility.', 'Use the "external" attribute for functions that are never called from within the contract.', 'Optimization']]

	def draw_pie(self, data=[], labels=[], use_colors=[]):
	    d = Drawing(500,230)
	    pie = Pie()
	    pie.x = 70 # x,y饼图在框中的坐标
	    pie.y = 5
	    pie.slices.label_boxStrokeColor = colors.white  #标签边框的颜色

	    pie.data = data      # 饼图上的数据
	    pie.labels = labels  # 数据的标签
	    pie.simpleLabels = 0 # 0 标签在标注线的右侧；1 在线上边
	    pie.sameRadii = 1    # 0 饼图是椭圆；1 饼图是圆形

	    pie.strokeWidth=0.5                         # 圆饼周围空白区域的宽度
	    pie.strokeColor= colors.white             # 整体饼图边界的颜色
	#     pie.slices.label_pointer_piePad = 10       # 圆饼和标签的距离
	#     pie.slices.label_pointer_edgePad = 25    # 标签和外边框的距离
	    pie.width = 180
	    pie.height = 180
	#     pie.direction = 'clockwise'
	    pie.pointerLabelMode  = 'LeftRight'
	#     print(dir(pie))
	    lg = Legend()
	    lg.x = 315
	    lg.y = 150
	    lg.dx = 20
	    lg.dy = 20
	    lg.deltax = 20
	    lg.deltay = 15
	    lg.dxTextSpace = 20
	    lg.columnMaximum = 6
	    lg.fontName = 'roman' #增加对中文字体的支持
	    lg.fontSize = 10.5
	    lg.colorNamePairs = list(zip(use_colors,labels))
	    lg.alignment = 'left'
	    lg.strokeColor = colors.white #legend边框颜色
	#     d.add(lab)
	    pie.slices.strokeColor = colors.white
	    pie.slices.strokeWidth = 0.5
	    for i in range(len(labels)):
	        pie.slices[i].fontName = 'roman' #设置中文
	        pie.slices[i].labelRadius = 0.6
	    for i, col in enumerate(use_colors):
	        pie.slices[i].fillColor  = col
	    lab = Label()
	    lab.x = 230  #x和y是文字的位置坐标
	    lab.y = 210
	    lab.setText('Vulnerability severity distribution map')
	    lab.fontName = 'hei' #增加对中文字体的支持
	#     lab.boxFillColor=colors.HexColor(0x330066)
	#     print(dir(lab))
	    lab.fontSize = 15
	    d.add(lab)    
	    d.add(lg)
	    d.add(pie)
	    d.background = Rect(0,0,448,230,strokeWidth=1,strokeColor="#868686",fillColor=None) #边框颜色
	    return d

	def draw_img(self, path):
		if os.path.exists(path):
			# print(path)
			img = Image(80, 0, 330, 250, path)
			d = Drawing(500, 250)
			# img = Image(path, width=75*mm, height=60*mm)       # 读取指定路径下的图片
			d.add(img)
			# img.drawWidth = 75*mm        # 设置图片的宽度
			# img.drawHeight = 60*mm       # 设置图片的高度
			# print(img)
			return d
		else:
			return None
	# def __init__(self):
	# 	"""
		
	# 	"""
		

	def _output(self, result_maps, filename, time_start_para, auditcontent, report_path, contracts_names, auditid_para, bytecodes, opcodes, ifmap, positions_results, filecontent_lists):
		global time_start, auditid
		time_start = time_start_para
		auditid = auditid_para

		story = []
		fig_index = 1
		table_index = 1
		# 首页内容
		# story.append(Macro('canvas.saveState()'))
		# story.append(Macro("canvas.drawImage(r'report\研究报告封面-正面.jpg',0,0,"+str(A4[0])+","+str(A4[1])+")"))
		# story.append(Macro('canvas.setFillColorRGB(255,255,255)'))
		# story.append(Macro('canvas.setFont("hei", 20)'))
		# story.append(Macro("canvas.drawString(177, 396, '编号：07072016329851')"))
		# story.append(Macro("canvas.drawString(177, 346, '日期：2020-10-22')"))
		# story.append(Macro('canvas.restoreState()'))
		story.append(PageBreak())
		#剩余页time.strftime(’%Y{y}%m{m}%d{d}%H{h}%M{f}%S{s}’).format(y=‘年’, m=‘月’, d=‘日’, h=‘时’, f=‘分’, s=‘秒’)
		story.append(Paragraph("0x01 Summary Information", self.title_style))
		if '月' in time.strftime('%b', time_start):
			story.append(Paragraph("The VulHunter (VH, for short) platform received this smart contract security audit application and audited the contract in "+month_convert[time.strftime('%b', time_start)]+" "+time.strftime('%Y', time_start)+".", self.content_style))
		else:
			story.append(Paragraph("The VulHunter (VH, for short) platform received this smart contract security audit application and audited the contract in "+time.strftime('%b', time_start)+" "+time.strftime('%Y', time_start)+".", self.content_style))
		story.append(Paragraph('It is necessary to declare that VH only issues this report in respect of facts that have occurred or existed before the issuance of this report, and undertakes corresponding responsibilities for this. For the facts that occur or exist in the future, VH is unable to judge the security status of its smart contract, and will not be responsible for it. The security audit analysis and other content made in this report are based on the documents and information provided to smart analysis team by the information provider as of the issuance of this report (referred to as "provided information"). VH hypothesis: There is no missing, tampered, deleted or concealed information in the mentioned information. If the information that has been mentioned is missing, tampered with, deleted, concealed or reflected does not match the actual situation, VulHunter shall not be liable for any losses and adverse effects caused thereby.', self.content_style))
		# 审计信息
		story.append(Spacer(1, 1.5 * mm))
		story.append(Paragraph("Table " + str(table_index) + " Contract audit information", self.table_title_style))
		table_index += 1

		contracts_names_str = ""
		if len(contracts_names) > 3:
			contracts_names_str = contracts_names[0] + "," + contracts_names[1] + "," + contracts_names[2] + ",..."
		else:
			contracts_names_str = ','.join(contracts_names)

		task_data = [['Project','Description'],['Contract name',contracts_names_str],['Contract type','Ethereum contract'],['Code language','Solidity'],['Contract files',filename.split('/')[-1]],['Contract address',''],['Auditors','VulHunter team'],['Audit time',time.strftime("%Y-%m-%d %H:%M:%S", time_start)],['Audit tool','VulHunter (VH)']]
		task_table = Table(task_data, colWidths=[83 * mm, 83 * mm], rowHeights=9 * mm, style=self.common_style)
		story.append(task_table)
		story.append(Spacer(1, 2 * mm))
		story.append(Paragraph("Table 1 shows the relevant information of this contract audit in detail. The details and results of the contract security audit will be introduced in detail below.", self.content_style))

		story.append(Paragraph("0x02 Contract Audit Results", self.title_style))
		story.append(Paragraph("2.1 Vulnerability Distribution", self.sub_title_style))
		story.append(Paragraph("The severity of vulnerabilities in this security audit is distributed according to the level of impact and confidence:", self.content_style))
		story.append(Paragraph("Table " + str(table_index) + " Overview of contract audit vulnerability distribution", self.table_title_style))
		table_index += 1

		loophole_distribute = {'High':0,'Medium':0,'Low':0,'Informational':0,'Optimization':0}
		result_number_color = {}
		for i in range(1,len(self.table_result)):
		    if self.table_result[i][1] in result_maps.keys():
		        loophole_distribute_val = {'High':0,'Medium':0,'Low':0,'Informational':0,'Optimization':0}
		        for v in result_maps[self.table_result[i][1]]['predict_labels']:
		        	if v == 1:
		        	    loophole_distribute_val[self.table_result[i][10]] = loophole_distribute_val[self.table_result[i][10]] + 1
		        numberimpact = ""
		        numberimpact_nocolor = ""
		        if loophole_distribute_val['High'] != 0:
		        	loophole_distribute['High'] = loophole_distribute['High'] + loophole_distribute_val['High']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#E61A1A">High:' + str(loophole_distribute_val['High']) + '</font>'
		        	    numberimpact_nocolor = 'High:' + str(loophole_distribute_val['High'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#E61A1A ">High:' + str(loophole_distribute_val['High']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\High:' + str(loophole_distribute_val['High'])
		        if loophole_distribute_val['Medium'] != 0:
		        	loophole_distribute['Medium'] = loophole_distribute['Medium'] + loophole_distribute_val['Medium']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#FF6600">Medium:' + str(loophole_distribute_val['Medium']) + '</font>'
		        	    numberimpact_nocolor = 'Medium:' + str(loophole_distribute_val['Medium'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#FF6600">Medium:' + str(loophole_distribute_val['Medium']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\Medium:' + str(loophole_distribute_val['Medium'])
		        if loophole_distribute_val['Low'] != 0:
		        	loophole_distribute['Low'] = loophole_distribute['Low'] + loophole_distribute_val['Low']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#DDB822">Low:' + str(loophole_distribute_val['Low']) + '</font>'
		        	    numberimpact_nocolor = 'Low:' + str(loophole_distribute_val['Low'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#DDB822">Low:' + str(loophole_distribute_val['Low']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\Low:' + str(loophole_distribute_val['Low'])
		        if loophole_distribute_val['Informational'] != 0:
		        	loophole_distribute['Informational'] = loophole_distribute['Informational'] + loophole_distribute_val['Informational']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#ff66ff">Info:' + str(loophole_distribute_val['Informational']) + '</font>'
		        	    numberimpact_nocolor = 'Info:' + str(loophole_distribute_val['Informational'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#ff66ff">Info:' + str(loophole_distribute_val['Informational']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\Info:' + str(loophole_distribute_val['Informational'])
		        if loophole_distribute_val['Optimization'] != 0:
		        	loophole_distribute['Optimization'] = loophole_distribute['Optimization'] + loophole_distribute_val['Optimization']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#22DDDD">Opt:' + str(loophole_distribute_val['Optimization']) + '</font>'
		        	    numberimpact_nocolor = 'Opt:' + str(loophole_distribute_val['Optimization'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#22DDDD">Opt:' + str(loophole_distribute_val['Optimization']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\Opt:' + str(loophole_distribute_val['Optimization'])
		        if numberimpact_nocolor != "":
		        	self.table_result[i][5] = numberimpact_nocolor
		        	result_number_color[self.table_result[i][1]] = numberimpact
		        else:
		        	result_number_color[self.table_result[i][1]] = '<font color="#2BD591">' + self.table_result[i][5]

		task_data_1 = [['Vulnerability security distribution'],['High','Medium','Low','Info','Opt'],[loophole_distribute['High'],loophole_distribute['Medium'],loophole_distribute['Low'],loophole_distribute['Informational'],loophole_distribute['Optimization']]]
		task_table_1 = Table(task_data_1, colWidths=[30 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm], rowHeights=9 * mm, style=self.common_style_1)
		story.append(task_table_1)
		pie_data = task_data_1[2]
		pie_labs = task_data_1[1]
		pie_color = [colors.HexColor('#E61A1A'),colors.HexColor('#FF6600'),colors.HexColor('#DDB822'),colors.HexColor('#ff66ff'),colors.HexColor('#22DDDD')]
		task_pie = self.draw_pie(pie_data,pie_labs,pie_color)
		story.append(Spacer(1, 1.5 * mm))
		story.append(task_pie)
		story.append(Paragraph("Figure " + str(fig_index) + " Vulnerability security distribution map", self.graph_title_style))
		fig_index += 1
		story.append(Paragraph("This security audit found "+str(loophole_distribute['High'])+" High-severity vulnerabilities, "+str(loophole_distribute['Medium'])+" Medium-severity vulnerabilities, "+str(loophole_distribute['Low'])+" Low-severity vulnerabilities, "+str(loophole_distribute['Optimization'])+" Optimization-severity vulnerabilities, and "+str(loophole_distribute['Informational'])+" places that need attention.", self.content_daoyin_style_red))
		story.append(Paragraph("2.2 Audit Results", self.sub_title_style))
		story.append(Paragraph("There are 31 test items in this security audit, and the test items are as follows (other unknown security vulnerabilities are not included in the scope of responsibility of this audit):", self.content_style))
		# common_style_result_all_type
		for i in range(1,len(self.table_result)):
		    if 'High' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#E61A1A')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#E61A1A')))
		    elif 'Medium' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#FF6600')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#FF6600')))
		    elif 'Low' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#DDB822')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#DDB822')))
		    elif 'Info' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#ff66ff')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#ff66ff')))
		    elif 'Opt' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#22DDDD')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#22DDDD')))
		    if i%2==1:
		        self.common_style_result_all_type.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#d9e2f3')))
		common_style_result_all = TableStyle(self.common_style_result_all_type)
		story.append(Paragraph("Table " + str(table_index) + " Contract audit items", self.table_title_style))
		table_index += 1
		task_table_2 = Table([var[0:6] for var in self.table_result], colWidths=[8 * mm, 47 * mm, 70 * mm, 15 * mm, 19 * mm, 20 * mm], rowHeights=7.5 * mm, style=common_style_result_all)
		story.append(task_table_2)

		story.append(Paragraph("0x03 Contract Code", self.title_style))
		story.append(Paragraph("3.1 Code", self.sub_title_style))
		# story.append(Paragraph("In the corresponding position of each contract code, security vulnerabilities and coding specification issues have been marked in the form of comments. The comment labels start with //StFt. For details, please refer to the following contract code content.", self.content_style))
		story.append(Paragraph(auditcontent, self.code_style))
		story.append(Paragraph("3.2 Extracted instances of the contract", self.sub_title_style))
		bytecodes_str = ''
		# print(bytecodes)
		for i in range(len(bytecodes)):
			bytecodes_str = bytecodes_str + 'The ' + str(i+1) + 'st instance:' + '<br/>' + ','.join([str(vv) for vv in bytecodes[i]]) + '<br/>'

		story.append(Paragraph(bytecodes_str, self.code_style))

		story.append(Paragraph("0x04 Contract Audit Details", self.title_style))
		for i in range(1,len(self.table_result)):
		    story.append(Paragraph('<font style="font-weight:bold">4.'+str(i)+' '+self.table_result[i][1]+'</font>', self.sub_title_style_romanbold))
		    if 'Pass' in self.table_result[i][5]:
		        #通过的
		        story.append(Paragraph("Vulnerability description", self.sub_sub_title_style))
		        story.append(Paragraph(self.table_result[i][6], self.content_style))
		        story.append(Paragraph("Applicable scenarios", self.sub_sub_title_style))
		        story.append(Paragraph(self.table_result[i][7], self.code_style))
		        if self.table_result[i][8]:
		        	story.append(Paragraph(self.table_result[i][8], self.content_style_codeadd))
		        story.append(Paragraph('Audit results:<font color="#2BD591">【Pass】</font>', self.sub_sub_title_style))
		        story.append(Paragraph('Security advice: none.', self.sub_sub_title_style))
		    else:
		        #有漏洞的
		        story.append(Paragraph("Vulnerability description", self.sub_sub_title_style))
		        story.append(Paragraph(self.table_result[i][6], self.content_style))
		        story.append(Paragraph('Applicable scenarios：【'+result_number_color[self.table_result[i][1]]+'】', self.sub_sub_title_style))
		        story.append(Paragraph("For this pattern, the specific problems in the contract are as follows:", self.content_style))

		        if ifmap == 'map':
		        	# story.append(Spacer(1, 1.5 * mm))
		        	# img_path = positions_results[self.table_result[i][1]][0]
		        	img = self.draw_img(positions_results[self.table_result[i][1]][0])
		        	# if os.path.exists(img_path):
		        		# f = open(img_path, 'rb')
		        		# img = Image(img_path, width=75*mm, height=60*mm)
		        		# img = flowable_fig(img_path)
		        	story.append(img)
		        		# story.append(Image(f, width=75*mm, height=60*mm))
		        		# del img
		        	story.append(Paragraph("Figure " + str(fig_index) + " The visualization  of  model detection for the vunlunbility " + self.table_result[i][1] + ".", self.graph_title_style))
		        	fig_index += 1

		        for ii in range(len(result_maps[self.table_result[i][1]]['predict_labels'])):
		        	if result_maps[self.table_result[i][1]]['predict_labels'][ii] == 0:
		        	    continue
		        	
		        	result_predict = result_maps[self.table_result[i][1]]['instances_predict_labels'][ii]
		        	description_vals = []
		        	description_vals.append('[' + ','.join([str(vv) for vv in result_predict]) + ']')
		        	for jj in range(len(result_predict)):
		        	    if result_predict[jj] == 1:
		        	        description_vals.append('[' + ','.join([str(vv) for vv in opcodes[jj]]) + ']')
		        	if self.table_result[i][10] == 'High':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#E61A1A" face="roman">(High):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	    # for j in range(1,len(description_vals)):
		        	    #     story.append(Paragraph('<font face="roman">The ' + str(j) + 'st suspected vulnerable execution sequence:</font> ' + description_vals[j], self.content_style_roman))
		        	elif self.table_result[i][10] == 'Medium':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#FF6600" face="roman">(Medium):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	    # for j in range(1,len(description_vals)):
		        	    #     story.append(Paragraph('<font face="roman">The ' + str(j) + 'st suspected vulnerable execution sequence:</font> ' + description_vals[j], self.content_style_roman))
		        	elif self.table_result[i][10] == 'Low':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#DDB822" face="roman">(Low):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	    # for j in range(1,len(description_vals)):
		        	    #     story.append(Paragraph('<font face="roman">The ' + str(j) + 'st suspected vulnerable execution sequence:</font> ' + description_vals[j], self.content_style_roman))
		        	elif self.table_result[i][10] == 'Informational':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#ff66ff" face="roman">(Informational):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	    # for j in range(1,len(description_vals)):
		        	    #     story.append(Paragraph('<font face="roman">The ' + str(j) + 'st suspected vulnerable execution sequence:</font> ' + description_vals[j], self.content_style_roman))              
		        	elif self.table_result[i][10] == 'Optimization':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#22DDDD" face="roman">(Optimization):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	    # for j in range(1,len(description_vals)):
		        	    #     story.append(Paragraph('<font face="roman">The ' + str(j) + 'st suspected vulnerable execution sequence:</font> ' + description_vals[j], self.content_style_roman))
		        	if ifmap == 'map':
		        		positions_results_val = positions_results[self.table_result[i][1]][1]
		        		for j in range(1,len(description_vals)):
		        			story.append(Paragraph('<font face="roman">The ' + str(j) + 'st suspected vulnerable execution sequence:</font> ' + description_vals[j], self.content_style_roman))
		        			story.append(Paragraph('<font face="roman">The model makes the decision may rely on the following slicens:</font>', self.content_style_roman))
		        			for iiii in range(len(positions_results_val[j-1][1])):
		        			    if positions_results_val[j-1][1][iiii][1]:
		        			    	position_val_str = "{}-{}".format(positions_results_val[j-1][1][iiii][2]['begin'], positions_results_val[j-1][1][iiii][2]['end'])
		        			    	story.append(Paragraph('  -  <font face="roman">The ' + str(iiii + 1) + ' -th position of suspected source code (importance: ' + str(positions_results_val[j-1][1][iiii][0]) + ', position: ' + str(positions_results_val[j-1][1][iiii][3]) + '): ' + position_val_str + '.</font>', self.content_style_roman))
		        			    	contract_content = ""
		        			    	if positions_results_val[j-1][1][iiii][2]['begin']['line'] == positions_results_val[j-1][1][iiii][2]['end']['line']:
		        			    		contract_content = filecontent_lists[positions_results_val[j-1][1][iiii][2]['begin']['line']][positions_results_val[j-1][1][iiii][2]['begin']['column']:(positions_results_val[j-1][1][iiii][2]['end']['column'] + 1)]
		        			    	else:
		        			    		contract_content = filecontent_lists[positions_results_val[j-1][1][iiii][2]['begin']['line']][positions_results_val[j-1][1][iiii][2]['begin']['column']:]
		        			    		contract_content += " ...... "
		        			    		contract_content += filecontent_lists[positions_results_val[j-1][1][iiii][2]['end']['line']][:(positions_results_val[j-1][1][iiii][2]['end']['column'] + 1)]
		        			    	story.append(Paragraph('  -   -   <font face="roman">The corresponding slicen of the source code is: ' + contract_content + '</font>', self.content_style_roman))
		        			    else:
		        			    	story.append(Paragraph('  -  <font face="roman">The ' + str(iiii + 1) + ' -th position of suspected source code (importance: ' + str(positions_results_val[j-1][1][iiii][0]) + ', position: ' + str(positions_results_val[j-1][1][iiii][3]) + '): ' + str(positions_results_val[j-1][1][iiii][2]) + '.</font>', self.content_style_roman))
		        	else:
		        		for j in range(1,len(description_vals)):
		        			story.append(Paragraph('<font face="roman">The ' + str(j) + 'st suspected vulnerable execution sequence:</font> ' + description_vals[j], self.content_style_roman))
		        story.append(Paragraph('Security advice', self.sub_sub_title_style))
		        story.append(Paragraph(self.table_result[i][9], self.content_style))
		# story.append(c)
		story.append(PageBreak())
		#尾页
		story.append(Spacer(0, 20 * mm))
		# story.append(Macro('canvas.saveState()'))
		# story.append(Macro("canvas.drawImage(r'report\研究报告封面-背面.jpg',0,0,"+str(A4[0])+","+str(A4[1])+")"))
		# story.append(Macro('canvas.restoreState()'))
		doc = SimpleDocTemplate(report_path,
		        	            pagesize=A4,
		        	            leftMargin=20 * mm, rightMargin=20 * mm, topMargin=27 * mm, bottomMargin=25 * mm)
		# print("1233333333333333333333333333333333")
		doc.build(story,canvasmaker=NumberedCanvasEnglish)
		print("The audit report has been saved to "+report_path+".")
	
	#output the main audit result
	def _output_main(self, result_maps, filename, time_start_para, auditcontent, report_path, contracts_names, auditid_para, bytecodes, opcodes, ifmap, positions_results, filecontent_lists):
		global time_start, auditid
		time_start = time_start_para
		auditid = auditid_para

		story = []
		fig_index = 1
		table_index = 1
		# 首页内容
		# story.append(Macro('canvas.saveState()'))
		# story.append(Macro("canvas.drawImage(r'report\研究报告封面-正面.jpg',0,0,"+str(A4[0])+","+str(A4[1])+")"))
		# story.append(Macro('canvas.setFillColorRGB(255,255,255)'))
		# story.append(Macro('canvas.setFont("hei", 20)'))
		# story.append(Macro("canvas.drawString(177, 396, '编号：07072016329851')"))
		# story.append(Macro("canvas.drawString(177, 346, '日期：2020-10-22')"))
		# story.append(Macro('canvas.restoreState()'))
		story.append(PageBreak())
		#剩余页time.strftime(’%Y{y}%m{m}%d{d}%H{h}%M{f}%S{s}’).format(y=‘年’, m=‘月’, d=‘日’, h=‘时’, f=‘分’, s=‘秒’)
		story.append(Paragraph("0x01 Summary Information", self.title_style))
		if '月' in time.strftime('%b', time_start):
			story.append(Paragraph("The VulHunter (VH, for short) platform received this smart contract security audit application and audited the contract in "+month_convert[time.strftime('%b', time_start)]+" "+time.strftime('%Y', time_start)+".", self.content_style))
		else:
			story.append(Paragraph("The VulHunter (VH, for short) platform received this smart contract security audit application and audited the contract in "+time.strftime('%b', time_start)+" "+time.strftime('%Y', time_start)+".", self.content_style))
		story.append(Paragraph('It is necessary to declare that VH only issues this report in respect of facts that have occurred or existed before the issuance of this report, and undertakes corresponding responsibilities for this. For the facts that occur or exist in the future, VH is unable to judge the security status of its smart contract, and will not be responsible for it. The security audit analysis and other content made in this report are based on the documents and information provided to smart analysis team by the information provider as of the issuance of this report (referred to as "provided information"). VH hypothesis: There is no missing, tampered, deleted or concealed information in the mentioned information. If the information that has been mentioned is missing, tampered with, deleted, concealed or reflected does not match the actual situation, VulHunter shall not be liable for any losses and adverse effects caused thereby.', self.content_style))
		# 审计信息
		story.append(Spacer(1, 1.5 * mm))
		story.append(Paragraph("Table " + str(table_index) + " Contract audit information", self.table_title_style))
		table_index += 1

		contracts_names_str = ""
		if len(contracts_names) > 3:
			contracts_names_str = contracts_names[0] + "," + contracts_names[1] + "," + contracts_names[2] + ",..."
		else:
			contracts_names_str = ','.join(contracts_names)

		task_data = [['Project','Description'],['Contract name',contracts_names_str],['Contract type','Ethereum contract'],['Code language','Solidity'],['Contract files',filename.split('/')[-1]],['Contract address',''],['Auditors','VulHunter team'],['Audit time',time.strftime("%Y-%m-%d %H:%M:%S", time_start)],['Audit tool','VulHunter (VH)']]
		task_table = Table(task_data, colWidths=[83 * mm, 83 * mm], rowHeights=9 * mm, style=self.common_style)
		story.append(task_table)
		story.append(Spacer(1, 2 * mm))
		story.append(Paragraph("Table 1 shows the relevant information of this contract audit in detail. The details and results of the contract security audit will be introduced in detail below.", self.content_style))

		story.append(Paragraph("0x02 Contract Audit Results", self.title_style))
		story.append(Paragraph("2.1 Vulnerability Distribution", self.sub_title_style))
		story.append(Paragraph("The severity of vulnerabilities in this security audit is distributed according to the level of impact and confidence:", self.content_style))
		story.append(Paragraph("Table " + str(table_index) + " Overview of contract audit vulnerability distribution", self.table_title_style))
		table_index += 1

		loophole_distribute = {'High':0,'Medium':0,'Low':0,'Informational':0,'Optimization':0}
		result_number_color = {}
		for i in range(1,len(self.table_result)):
		    if self.table_result[i][1] in result_maps.keys():
		        loophole_distribute_val = {'High':0,'Medium':0,'Low':0,'Informational':0,'Optimization':0}
		        for v in result_maps[self.table_result[i][1]]['predict_labels']:
		        	if v == 1:
		        	    loophole_distribute_val[self.table_result[i][10]] = loophole_distribute_val[self.table_result[i][10]] + 1
		        numberimpact = ""
		        numberimpact_nocolor = ""
		        if loophole_distribute_val['High'] != 0:
		        	loophole_distribute['High'] = loophole_distribute['High'] + loophole_distribute_val['High']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#E61A1A">High:' + str(loophole_distribute_val['High']) + '</font>'
		        	    numberimpact_nocolor = 'High:' + str(loophole_distribute_val['High'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#E61A1A ">High:' + str(loophole_distribute_val['High']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\High:' + str(loophole_distribute_val['High'])
		        if loophole_distribute_val['Medium'] != 0:
		        	loophole_distribute['Medium'] = loophole_distribute['Medium'] + loophole_distribute_val['Medium']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#FF6600">Medium:' + str(loophole_distribute_val['Medium']) + '</font>'
		        	    numberimpact_nocolor = 'Medium:' + str(loophole_distribute_val['Medium'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#FF6600">Medium:' + str(loophole_distribute_val['Medium']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\Medium:' + str(loophole_distribute_val['Medium'])
		        if loophole_distribute_val['Low'] != 0:
		        	loophole_distribute['Low'] = loophole_distribute['Low'] + loophole_distribute_val['Low']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#DDB822">Low:' + str(loophole_distribute_val['Low']) + '</font>'
		        	    numberimpact_nocolor = 'Low:' + str(loophole_distribute_val['Low'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#DDB822">Low:' + str(loophole_distribute_val['Low']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\Low:' + str(loophole_distribute_val['Low'])
		        if loophole_distribute_val['Informational'] != 0:
		        	loophole_distribute['Informational'] = loophole_distribute['Informational'] + loophole_distribute_val['Informational']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#ff66ff">Info:' + str(loophole_distribute_val['Informational']) + '</font>'
		        	    numberimpact_nocolor = 'Info:' + str(loophole_distribute_val['Informational'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#ff66ff">Info:' + str(loophole_distribute_val['Informational']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\Info:' + str(loophole_distribute_val['Informational'])
		        if loophole_distribute_val['Optimization'] != 0:
		        	loophole_distribute['Optimization'] = loophole_distribute['Optimization'] + loophole_distribute_val['Optimization']
		        	if numberimpact == '':
		        	    numberimpact = '<font color="#22DDDD">Opt:' + str(loophole_distribute_val['Optimization']) + '</font>'
		        	    numberimpact_nocolor = 'Opt:' + str(loophole_distribute_val['Optimization'])
		        	else:
		        	    numberimpact = numberimpact + '\\<font color="#22DDDD">Opt:' + str(loophole_distribute_val['Optimization']) + '</font>'
		        	    numberimpact_nocolor = numberimpact_nocolor + '\\Opt:' + str(loophole_distribute_val['Optimization'])
		        if numberimpact_nocolor != "":
		        	self.table_result[i][5] = numberimpact_nocolor
		        	result_number_color[self.table_result[i][1]] = numberimpact
		        else:
		        	result_number_color[self.table_result[i][1]] = '<font color="#2BD591">' + self.table_result[i][5]

		task_data_1 = [['Vulnerability security distribution'],['High','Medium','Low','Info','Opt'],[loophole_distribute['High'],loophole_distribute['Medium'],loophole_distribute['Low'],loophole_distribute['Informational'],loophole_distribute['Optimization']]]
		task_table_1 = Table(task_data_1, colWidths=[30 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm], rowHeights=9 * mm, style=self.common_style_1)
		story.append(task_table_1)
		pie_data = task_data_1[2]
		pie_labs = task_data_1[1]
		pie_color = [colors.HexColor('#E61A1A'),colors.HexColor('#FF6600'),colors.HexColor('#DDB822'),colors.HexColor('#ff66ff'),colors.HexColor('#22DDDD')]
		task_pie = self.draw_pie(pie_data,pie_labs,pie_color)
		story.append(Spacer(1, 1.5 * mm))
		story.append(task_pie)
		story.append(Paragraph("Figure " + str(fig_index) + " Vulnerability security distribution map", self.graph_title_style))
		fig_index += 1
		story.append(Paragraph("This security audit found "+str(loophole_distribute['High'])+" High-severity vulnerabilities, "+str(loophole_distribute['Medium'])+" Medium-severity vulnerabilities, "+str(loophole_distribute['Low'])+" Low-severity vulnerabilities, "+str(loophole_distribute['Optimization'])+" Optimization-severity vulnerabilities, and "+str(loophole_distribute['Informational'])+" places that need attention.", self.content_daoyin_style_red))
		story.append(Paragraph("2.2 Audit Results", self.sub_title_style))
		story.append(Paragraph("There are 31 test items in this security audit, and the test items are as follows (other unknown security vulnerabilities are not included in the scope of responsibility of this audit):", self.content_style))
		# common_style_result_all_type
		for i in range(1,len(self.table_result)):
		    if 'High' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#E61A1A')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#E61A1A')))
		    elif 'Medium' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#FF6600')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#FF6600')))
		    elif 'Low' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#DDB822')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#DDB822')))
		    elif 'Info' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#ff66ff')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#ff66ff')))
		    elif 'Opt' in self.table_result[i][3]:
		        self.common_style_result_all_type.append(('TEXTCOLOR', (3, i), (3, i), colors.HexColor('#22DDDD')))
		        if 'Pass' in self.table_result[i][5]:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#2BD591')))
		        else:
		        	self.common_style_result_all_type.append(('TEXTCOLOR', (5, i), (5, i), colors.HexColor('#22DDDD')))
		    if i%2==1:
		        self.common_style_result_all_type.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#d9e2f3')))
		common_style_result_all = TableStyle(self.common_style_result_all_type)
		story.append(Paragraph("Table " + str(table_index) + " Contract audit items", self.table_title_style))
		table_index += 1
		task_table_2 = Table([var[0:6] for var in self.table_result], colWidths=[8 * mm, 47 * mm, 70 * mm, 15 * mm, 19 * mm, 20 * mm], rowHeights=7.5 * mm, style=common_style_result_all)
		story.append(task_table_2)

		story.append(Paragraph("0x03 Contract Code", self.title_style))
		story.append(Paragraph("3.1 Code", self.sub_title_style))
		# story.append(Paragraph("In the corresponding position of each contract code, security vulnerabilities and coding specification issues have been marked in the form of comments. The comment labels start with //StFt. For details, please refer to the following contract code content.", self.content_style))
		story.append(Paragraph(auditcontent, self.code_style))
		story.append(Paragraph("3.2 Extracted instances of the contract", self.sub_title_style))
		bytecodes_str = ''
		# print(bytecodes)
		for i in range(len(bytecodes)):
			bytecodes_str = bytecodes_str + 'The ' + str(i+1) + 'st instance:' + '<br/>' + ','.join([str(vv) for vv in bytecodes[i]]) + '<br/>'

		story.append(Paragraph(bytecodes_str, self.code_style))

		story.append(Paragraph("0x04 Contract Audit Details", self.title_style))
		num = 1
		for i in range(1,len(self.table_result)):
		    if 'Pass' not in self.table_result[i][5]:
		        #有漏洞的
		       	story.append(Paragraph('<font style="font-weight:bold">4.'+str(num)+' '+self.table_result[i][1]+'</font>', self.sub_title_style_romanbold))
		        story.append(Paragraph("Vulnerability description", self.sub_sub_title_style))
		        story.append(Paragraph(self.table_result[i][6], self.content_style))
		        story.append(Paragraph('Audit results: 【'+result_number_color[self.table_result[i][1]]+'】', self.sub_sub_title_style))
		        story.append(Paragraph("For this pattern, the specific problems in the contract are as follows: ", self.content_style))
		        
		        if ifmap == 'map':
		        	# story.append(Spacer(1, 1.5 * mm))
		        	# img_path = positions_results[self.table_result[i][1]][0]
		        	img = self.draw_img(positions_results[self.table_result[i][1]][0])
		        	# if os.path.exists(img_path):
		        		# f = open(img_path, 'rb')
		        		# img = Image(img_path, width=75*mm, height=60*mm)
		        		# img = flowable_fig(img_path)
		        	story.append(img)
		        		# story.append(Image(f, width=75*mm, height=60*mm))
		        		# del img
		        	story.append(Paragraph("Figure " + str(fig_index) + " The visualization  of  model detection for the vunlunbility " + self.table_result[i][1] + ".", self.graph_title_style))
		        	fig_index += 1
				
		        for ii in range(len(result_maps[self.table_result[i][1]]['predict_labels'])):
		        	if result_maps[self.table_result[i][1]]['predict_labels'][ii] == 0:
		        		continue
		        	
		        	result_predict = result_maps[self.table_result[i][1]]['instances_predict_labels'][ii]
		        	
		        	description_vals = []
		        	description_vals.append('[' + ','.join([str(vv) for vv in result_predict]) + ']')
		        	for jj in range(len(result_predict)):
		        	    if result_predict[jj] == 1:
		        	        description_vals.append('[' + ','.join([str(vv) for vv in opcodes[jj]]) + ']')
		        	if self.table_result[i][10] == 'High':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#E61A1A" face="roman">(High):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	elif self.table_result[i][10] == 'Medium':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#FF6600" face="roman">(Medium):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	elif self.table_result[i][10] == 'Low':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#DDB822" face="roman">(Low):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	elif self.table_result[i][10] == 'Informational':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#ff66ff" face="roman">(Informational):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	elif self.table_result[i][10] == 'Optimization':
		        	    # description_vals = v['description'].replace("\t","&#160;&#160;&#160;&#160;").split('\n')
		        	    story.append(Paragraph('<font color="#22DDDD" face="roman">(Optimization):</font>'+description_vals[0] + '. <font face="roman">The vulnerability is detailed as follows.</font>', self.content_style_roman))
		        	   
		        	if ifmap == 'map':
		        		positions_results_val = positions_results[self.table_result[i][1]][1]
		        		for j in range(1,len(description_vals)):
		        			story.append(Paragraph('<font face="roman">The ' + str(j) + 'st suspected vulnerable execution sequence:</font> ' + description_vals[j], self.content_style_roman))
		        			story.append(Paragraph('<font face="roman">The model makes the decision may rely on the following slicens:</font>', self.content_style_roman))
		        			for iiii in range(len(positions_results_val[j-1][1])):
		        				if positions_results_val[j-1][1][iiii][1]:
		        					position_val_str = "{}-{}".format(positions_results_val[j-1][1][iiii][2]['begin'], positions_results_val[j-1][1][iiii][2]['end'])
		        					story.append(Paragraph('  -  <font face="roman">The ' + str(iiii + 1) + ' -th position of suspected source code (importance: ' + str(positions_results_val[j-1][1][iiii][0]) + ', position: ' + str(positions_results_val[j-1][1][iiii][3]) + '): ' + position_val_str + '.</font>', self.content_style_roman))
		        					contract_content = ""
		        					if positions_results_val[j-1][1][iiii][2]['begin']['line'] == positions_results_val[j-1][1][iiii][2]['end']['line']:
		        						contract_content = filecontent_lists[positions_results_val[j-1][1][iiii][2]['begin']['line']][positions_results_val[j-1][1][iiii][2]['begin']['column']:(positions_results_val[j-1][1][iiii][2]['end']['column'] + 1)]
		        					else:
		        						contract_content = filecontent_lists[positions_results_val[j-1][1][iiii][2]['begin']['line']][positions_results_val[j-1][1][iiii][2]['begin']['column']:]
		        						contract_content += " ...... "
		        						contract_content += filecontent_lists[positions_results_val[j-1][1][iiii][2]['end']['line']][:(positions_results_val[j-1][1][iiii][2]['end']['column'] + 1)]
		        					story.append(Paragraph('  -   -   <font face="roman">The corresponding slicen of the source code is: ' + contract_content + '</font>', self.content_style_roman))
		        				else:
		        					story.append(Paragraph('  -  <font face="roman">The ' + str(iiii + 1) + ' -th position of suspected source code (importance: ' + str(positions_results_val[j-1][1][iiii][0]) + ', position: ' + str(positions_results_val[j-1][1][iiii][3]) + '): ' + str(positions_results_val[j-1][1][iiii][2]) + '.</font>', self.content_style_roman))
		        	else:
		        		for j in range(1,len(description_vals)):
		        			story.append(Paragraph('<font face="roman">The ' + str(j) + 'st suspected vulnerable execution sequence:</font> ' + description_vals[j], self.content_style_roman))
		        story.append(Paragraph('Security advice', self.sub_sub_title_style))
		        story.append(Paragraph(self.table_result[i][9], self.content_style))
		        num = num + 1
		# story.append(c)
		story.append(PageBreak())
		#尾页
		story.append(Spacer(0, 20 * mm))
		# story.append(Macro('canvas.saveState()'))
		# story.append(Macro("canvas.drawImage(r'report\研究报告封面-背面.jpg',0,0,"+str(A4[0])+","+str(A4[1])+")"))
		# story.append(Macro('canvas.restoreState()'))
		doc = SimpleDocTemplate(report_path,
		        	            pagesize=A4,
		        	            leftMargin=20 * mm, rightMargin=20 * mm, topMargin=27 * mm, bottomMargin=25 * mm)
		# print("1233333333333333333333333333333333")
		doc.build(story,canvasmaker=NumberedCanvasEnglish)
		print("The audit report has been saved to "+report_path+".")