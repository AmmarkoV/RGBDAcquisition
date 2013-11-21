#include "SelectSegmentation.h"

#include <wx/msgdlg.h>
//(*InternalHeaders(SelectSegmentation)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(SelectSegmentation)
const long SelectSegmentation::ID_STATICBOX2 = wxNewId();
const long SelectSegmentation::ID_BUTTON1 = wxNewId();
const long SelectSegmentation::ID_BUTTON2 = wxNewId();
const long SelectSegmentation::ID_STATICBOX1 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT1 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL1 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL2 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL3 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT2 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL4 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL5 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL6 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT3 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT4 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT5 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT6 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL1 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL2 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT7 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL3 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL4 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT8 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL7 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT9 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL8 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT10 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL5 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL6 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT11 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL7 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL8 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL9 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL10 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL11 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL12 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL13 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL14 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT13 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT14 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT16 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL15 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL16 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL17 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT17 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL18 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL19 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL20 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL21 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL22 = wxNewId();
const long SelectSegmentation::ID_TEXTCTRL23 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT18 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT19 = wxNewId();
const long SelectSegmentation::ID_CHOICE1 = wxNewId();
const long SelectSegmentation::ID_CHECKBOX1 = wxNewId();
const long SelectSegmentation::ID_CHECKBOX2 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT12 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL9 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL10 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL11 = wxNewId();
const long SelectSegmentation::ID_STATICTEXT15 = wxNewId();
const long SelectSegmentation::ID_SPINCTRL12 = wxNewId();
const long SelectSegmentation::ID_STATICLINE1 = wxNewId();
const long SelectSegmentation::ID_BUTTON3 = wxNewId();
//*)

BEGIN_EVENT_TABLE(SelectSegmentation,wxDialog)
	//(*EventTable(SelectSegmentation)
	//*)
END_EVENT_TABLE()

SelectSegmentation::SelectSegmentation(wxWindow* parent,wxWindowID id)
{
	//(*Initialize(SelectSegmentation)
	Create(parent, id, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(748,549));
	StaticBox2 = new wxStaticBox(this, ID_STATICBOX2, _("Depth"), wxPoint(376,16), wxSize(352,400), 0, _T("ID_STATICBOX2"));
	ButtonCancel = new wxButton(this, ID_BUTTON1, _("Cancel"), wxPoint(632,456), wxSize(85,56), 0, wxDefaultValidator, _T("ID_BUTTON1"));
	ButtonOk = new wxButton(this, ID_BUTTON2, _("Ok"), wxPoint(376,456), wxSize(248,56), 0, wxDefaultValidator, _T("ID_BUTTON2"));
	StaticBox1 = new wxStaticBox(this, ID_STATICBOX1, _("RGB"), wxPoint(16,16), wxSize(352,400), 0, _T("ID_STATICBOX1"));
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Minimum : "), wxPoint(40,66), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	minR = new wxSpinCtrl(this, ID_SPINCTRL1, _T("0"), wxPoint(128,64), wxSize(56,23), 0, 0, 255, 0, _T("ID_SPINCTRL1"));
	minR->SetValue(_T("0"));
	minG = new wxSpinCtrl(this, ID_SPINCTRL2, _T("0"), wxPoint(200,64), wxSize(56,23), 0, 0, 100, 0, _T("ID_SPINCTRL2"));
	minG->SetValue(_T("0"));
	minB = new wxSpinCtrl(this, ID_SPINCTRL3, _T("0"), wxPoint(272,64), wxSize(56,23), 0, 0, 100, 0, _T("ID_SPINCTRL3"));
	minB->SetValue(_T("0"));
	StaticText2 = new wxStaticText(this, ID_STATICTEXT2, _("Maximum :"), wxPoint(40,100), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	maxR = new wxSpinCtrl(this, ID_SPINCTRL4, _T("255"), wxPoint(128,96), wxSize(56,23), 0, 0, 255, 255, _T("ID_SPINCTRL4"));
	maxR->SetValue(_T("255"));
	maxG = new wxSpinCtrl(this, ID_SPINCTRL5, _T("255"), wxPoint(200,96), wxSize(56,23), 0, 0, 255, 255, _T("ID_SPINCTRL5"));
	maxG->SetValue(_T("255"));
	maxB = new wxSpinCtrl(this, ID_SPINCTRL6, _T("255"), wxPoint(272,96), wxSize(56,23), 0, 0, 255, 255, _T("ID_SPINCTRL6"));
	maxB->SetValue(_T("255"));
	StaticText3 = new wxStaticText(this, ID_STATICTEXT3, _("R"), wxPoint(150,40), wxDefaultSize, 0, _T("ID_STATICTEXT3"));
	StaticText4 = new wxStaticText(this, ID_STATICTEXT4, _("G"), wxPoint(225,40), wxDefaultSize, 0, _T("ID_STATICTEXT4"));
	StaticText5 = new wxStaticText(this, ID_STATICTEXT5, _("B"), wxPoint(294,40), wxDefaultSize, 0, _T("ID_STATICTEXT5"));
	StaticText6 = new wxStaticText(this, ID_STATICTEXT6, _("Crop:"), wxPoint(40,160), wxDefaultSize, 0, _T("ID_STATICTEXT6"));
	cropRGBX1 = new wxTextCtrl(this, ID_TEXTCTRL1, _("0"), wxPoint(80,156), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
	cropRGBY1 = new wxTextCtrl(this, ID_TEXTCTRL2, _("0"), wxPoint(130,156), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL2"));
	StaticText7 = new wxStaticText(this, ID_STATICTEXT7, _("---->"), wxPoint(188,158), wxDefaultSize, 0, _T("ID_STATICTEXT7"));
	cropRGBX2 = new wxTextCtrl(this, ID_TEXTCTRL3, _("640"), wxPoint(232,156), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL3"));
	cropRGBY2 = new wxTextCtrl(this, ID_TEXTCTRL4, _("480"), wxPoint(284,156), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL4"));
	StaticText8 = new wxStaticText(this, ID_STATICTEXT8, _("Minimum :"), wxPoint(392,66), wxDefaultSize, 0, _T("ID_STATICTEXT8"));
	minDepth = new wxSpinCtrl(this, ID_SPINCTRL7, _T("0"), wxPoint(480,64), wxDefaultSize, 0, 0, 10000, 0, _T("ID_SPINCTRL7"));
	minDepth->SetValue(_T("0"));
	StaticText9 = new wxStaticText(this, ID_STATICTEXT9, _("Maximum :"), wxPoint(392,100), wxDefaultSize, 0, _T("ID_STATICTEXT9"));
	maxDepth = new wxSpinCtrl(this, ID_SPINCTRL8, _T("10000"), wxPoint(480,96), wxDefaultSize, 0, 0, 10000, 10000, _T("ID_SPINCTRL8"));
	maxDepth->SetValue(_T("10000"));
	StaticText10 = new wxStaticText(this, ID_STATICTEXT10, _("Crop:"), wxPoint(392,160), wxDefaultSize, 0, _T("ID_STATICTEXT10"));
	cropDepthX1 = new wxTextCtrl(this, ID_TEXTCTRL5, _("0"), wxPoint(432,156), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL5"));
	cropDepthY1 = new wxTextCtrl(this, ID_TEXTCTRL6, _("0"), wxPoint(482,156), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL6"));
	StaticText11 = new wxStaticText(this, ID_STATICTEXT11, _("---->"), wxPoint(536,158), wxDefaultSize, 0, _T("ID_STATICTEXT11"));
	cropDepthX2 = new wxTextCtrl(this, ID_TEXTCTRL7, _("640"), wxPoint(584,156), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL7"));
	cropDepthY2 = new wxTextCtrl(this, ID_TEXTCTRL8, _("480"), wxPoint(640,156), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL8"));
	bboxMinX = new wxTextCtrl(this, ID_TEXTCTRL9, _("0.0"), wxPoint(496,224), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL9"));
	bboxMinY = new wxTextCtrl(this, ID_TEXTCTRL10, _("0.0"), wxPoint(560,224), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL10"));
	bboxMinZ = new wxTextCtrl(this, ID_TEXTCTRL11, _("0.0"), wxPoint(624,224), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL11"));
	bboxMaxX = new wxTextCtrl(this, ID_TEXTCTRL12, _("0.0"), wxPoint(496,256), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL12"));
	bboxMaxY = new wxTextCtrl(this, ID_TEXTCTRL13, _("0.0"), wxPoint(560,256), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL13"));
	bboxMaxZ = new wxTextCtrl(this, ID_TEXTCTRL14, _("0.0"), wxPoint(624,256), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL14"));
	StaticText13 = new wxStaticText(this, ID_STATICTEXT13, _("Minimum"), wxPoint(424,228), wxDefaultSize, 0, _T("ID_STATICTEXT13"));
	StaticText14 = new wxStaticText(this, ID_STATICTEXT14, _("Maximum"), wxPoint(424,258), wxDefaultSize, 0, _T("ID_STATICTEXT14"));
	StaticText16 = new wxStaticText(this, ID_STATICTEXT16, _("Point A"), wxPoint(424,312), wxDefaultSize, 0, _T("ID_STATICTEXT16"));
	planeP1X = new wxTextCtrl(this, ID_TEXTCTRL15, _("0.0"), wxPoint(496,308), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL15"));
	planeP1Y = new wxTextCtrl(this, ID_TEXTCTRL16, _("0.0"), wxPoint(560,308), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL16"));
	planeP1Z = new wxTextCtrl(this, ID_TEXTCTRL17, _("0.0"), wxPoint(624,308), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL17"));
	StaticText17 = new wxStaticText(this, ID_STATICTEXT17, _("Point B"), wxPoint(424,340), wxDefaultSize, 0, _T("ID_STATICTEXT17"));
	planeP2X = new wxTextCtrl(this, ID_TEXTCTRL18, _("0.0"), wxPoint(496,336), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL18"));
	planeP2Y = new wxTextCtrl(this, ID_TEXTCTRL19, _("0.0"), wxPoint(560,336), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL19"));
	planeP2Z = new wxTextCtrl(this, ID_TEXTCTRL20, _("0.0"), wxPoint(624,336), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL20"));
	planeP3X = new wxTextCtrl(this, ID_TEXTCTRL21, _("0.0"), wxPoint(496,364), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL21"));
	planeP3Y = new wxTextCtrl(this, ID_TEXTCTRL22, _("0.0"), wxPoint(560,364), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL22"));
	planeP3Z = new wxTextCtrl(this, ID_TEXTCTRL23, _("0.0"), wxPoint(624,364), wxSize(56,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL23"));
	StaticText18 = new wxStaticText(this, ID_STATICTEXT18, _("Point C"), wxPoint(424,368), wxDefaultSize, 0, _T("ID_STATICTEXT18"));
	StaticText19 = new wxStaticText(this, ID_STATICTEXT19, _("Combination Method"), wxPoint(16,460), wxDefaultSize, 0, _T("ID_STATICTEXT19"));
	ChoiceCombination = new wxChoice(this, ID_CHOICE1, wxPoint(168,456), wxSize(168,25), 0, 0, 0, wxDefaultValidator, _T("ID_CHOICE1"));
	ChoiceCombination->SetSelection( ChoiceCombination->Append(_("No combination")) );
	ChoiceCombination->Append(_("AND combination"));
	ChoiceCombination->Append(_("OR combination"));
	ChoiceCombination->Append(_("XOR combination "));
	ChoiceCombination->Append(_("RGB only"));
	ChoiceCombination->Append(_("Depth only "));
	CheckBoxBoundingBox = new wxCheckBox(this, ID_CHECKBOX1, _("Bounding Box"), wxPoint(392,204), wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX1"));
	CheckBoxBoundingBox->SetValue(false);
	CheckBoxPlane = new wxCheckBox(this, ID_CHECKBOX2, _("Plane Segmentation"), wxPoint(392,290), wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX2"));
	CheckBoxPlane->SetValue(false);
	StaticText12 = new wxStaticText(this, ID_STATICTEXT12, _("Erase Color is "), wxPoint(16,424), wxDefaultSize, 0, _T("ID_STATICTEXT12"));
	eraseColorR = new wxSpinCtrl(this, ID_SPINCTRL9, _T("255"), wxPoint(124,420), wxSize(48,23), 0, 0, 255, 255, _T("ID_SPINCTRL9"));
	eraseColorR->SetValue(_T("255"));
	eraseColorG = new wxSpinCtrl(this, ID_SPINCTRL10, _T("255"), wxPoint(176,420), wxSize(48,23), 0, 0, 255, 255, _T("ID_SPINCTRL10"));
	eraseColorG->SetValue(_T("255"));
	eraseColorB = new wxSpinCtrl(this, ID_SPINCTRL11, _T("255"), wxPoint(232,420), wxSize(48,23), 0, 0, 255, 255, _T("ID_SPINCTRL11"));
	eraseColorB->SetValue(_T("255"));
	StaticText15 = new wxStaticText(this, ID_STATICTEXT15, _("Erase Depth is "), wxPoint(384,424), wxDefaultSize, 0, _T("ID_STATICTEXT15"));
	eraseDepth = new wxSpinCtrl(this, ID_SPINCTRL12, _T("0"), wxPoint(496,420), wxDefaultSize, 0, 0, 10000, 0, _T("ID_SPINCTRL12"));
	eraseDepth->SetValue(_T("0"));
	StaticLine1 = new wxStaticLine(this, ID_STATICLINE1, wxPoint(20,450), wxSize(700,0), wxLI_HORIZONTAL, _T("ID_STATICLINE1"));
	ButtonExport = new wxButton(this, ID_BUTTON3, _("Export To File"), wxPoint(16,480), wxSize(120,27), 0, wxDefaultValidator, _T("ID_BUTTON3"));
	FileDialogExport = new wxFileDialog(this, _("Export Segmentation To File"), wxEmptyString, wxEmptyString, _(".txt"), wxFD_DEFAULT_STYLE|wxFD_SAVE, wxDefaultPosition, wxDefaultSize, _T("wxFileDialog"));

	Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectSegmentation::OnButtonCancelClick);
	Connect(ID_BUTTON2,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectSegmentation::OnButtonOkClick);
	Connect(ID_BUTTON3,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectSegmentation::OnButtonExportClick);
	//*)

	reloadSegmentationFormFromValues();
}

SelectSegmentation::~SelectSegmentation()
{
	//(*Destroy(SelectSegmentation)
	//*)
}


int SelectSegmentation::reloadSegmentationFormFromValues()
{
  wxString val;

  val.Clear(); val<<selectedDepthConf.minDepth;  minDepth->SetValue(val);
  val.Clear(); val<<selectedDepthConf.maxDepth;  maxDepth->SetValue(val);

  CheckBoxBoundingBox->SetValue( (selectedDepthConf.enableBBox!=0) );
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.bboxX1); bboxMinX->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.bboxY1); bboxMinY->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.bboxZ1); bboxMinZ->SetValue(val);

  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.bboxX2); bboxMaxX->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.bboxY2); bboxMaxY->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.bboxZ2); bboxMaxZ->SetValue(val);


  CheckBoxPlane->SetValue( (selectedDepthConf.enablePlaneSegmentation!=0) );
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.p1[0]); planeP1X->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.p1[1]); planeP1Y->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.p1[2]); planeP1Z->SetValue(val);

  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.p2[0]); planeP2X->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.p2[1]); planeP2Y->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.p2[2]); planeP2Z->SetValue(val);

  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.p3[0]); planeP3X->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.p3[1]); planeP3Y->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.2f"),selectedDepthConf.p3[2]); planeP3Z->SetValue(val);

   val.Clear(); val.Printf(wxT("%u"),selectedDepthConf.minX); cropDepthX1->SetValue(val);
   val.Clear(); val.Printf(wxT("%u"),selectedDepthConf.minY); cropDepthY1->SetValue(val);
   val.Clear(); val.Printf(wxT("%u"),selectedDepthConf.maxX); cropDepthX2->SetValue(val);
   val.Clear(); val.Printf(wxT("%u"),selectedDepthConf.maxY); cropDepthY2->SetValue(val);



   val.Clear(); val.Printf(wxT("%u"),selectedRGBConf.minX); cropRGBX1->SetValue(val);
   val.Clear(); val.Printf(wxT("%u"),selectedRGBConf.minY); cropRGBY1->SetValue(val);
   val.Clear(); val.Printf(wxT("%u"),selectedRGBConf.maxX); cropRGBX2->SetValue(val);
   val.Clear(); val.Printf(wxT("%u"),selectedRGBConf.maxY); cropRGBY2->SetValue(val);

   minR->SetValue( selectedRGBConf.minR );
   minG->SetValue(selectedRGBConf.minG);
   minB->SetValue(selectedRGBConf.minB);
   maxR->SetValue(selectedRGBConf.maxR);
   maxG->SetValue(selectedRGBConf.maxG );
   maxB->SetValue(selectedRGBConf.maxB);



  ChoiceCombination->SetSelection(selectedCombinationMode);

  Refresh();

  return 1;
}


int SelectSegmentation::saveSegmentationValuesFromForm()
{
   float fValue;
   double dValue;
   long value;

   if (CheckBoxBoundingBox->IsChecked())
   {
    if (this->bboxMinX->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.bboxX1 = dValue; }
    if (this->bboxMinY->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.bboxY1 = dValue; }
    if (this->bboxMinZ->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.bboxZ1 = dValue; }

    if (this->bboxMaxX->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.bboxX2 = dValue; }
    if (this->bboxMaxY->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.bboxY2 = dValue; }
    if (this->bboxMaxZ->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.bboxZ2 = dValue; }

    selectedDepthConf.enableBBox=1;
   }

   if (CheckBoxPlane->IsChecked())
   {
    if (this->planeP1X->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.p1[0] = dValue; }
    if (this->planeP1Y->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.p1[1] = dValue; }
    if (this->planeP1Z->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.p1[2] = dValue; }

    if (this->planeP2X->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.p2[0] = dValue; }
    if (this->planeP2Y->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.p2[1] = dValue; }
    if (this->planeP2Z->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.p2[2] = dValue; }

    if (this->planeP3X->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.p3[0] = dValue; }
    if (this->planeP3Y->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.p3[1] = dValue; }
    if (this->planeP3Z->GetValue().ToDouble(&dValue)) {  this->selectedDepthConf.p3[2] = dValue; }

    selectedDepthConf.enablePlaneSegmentation=1;
   }

  if (cropDepthX1->GetValue().ToLong(&value)) {  selectedDepthConf.minX = value; }
  if (cropDepthY1->GetValue().ToLong(&value)) {  selectedDepthConf.minY = value; }
  if (cropDepthX2->GetValue().ToLong(&value)) {  selectedDepthConf.maxX = value; }
  if (cropDepthY2->GetValue().ToLong(&value)) {  selectedDepthConf.maxY = value; }

  if (cropRGBX1->GetValue().ToLong(&value)) {  selectedRGBConf.minX = value; }
  if (cropRGBY1->GetValue().ToLong(&value)) {  selectedRGBConf.minY = value; }
  if (cropRGBX2->GetValue().ToLong(&value)) {  selectedRGBConf.maxX = value; }
  if (cropRGBY2->GetValue().ToLong(&value)) {  selectedRGBConf.maxY = value; }

  selectedDepthConf.minDepth = minDepth->GetValue();
  selectedDepthConf.maxDepth = maxDepth->GetValue();

  //selectedDepthConf.



   selectedRGBConf.minR = minR->GetValue();
   selectedRGBConf.minG = minG->GetValue();
   selectedRGBConf.minB = minB->GetValue();
   selectedRGBConf.maxR = maxR->GetValue();
   selectedRGBConf.maxG = maxG->GetValue();
   selectedRGBConf.maxB = maxB->GetValue();


   selectedRGBConf.eraseColorR = eraseColorR->GetValue();
   selectedRGBConf.eraseColorG = eraseColorG->GetValue();
   selectedRGBConf.eraseColorB = eraseColorB->GetValue();




  selectedCombinationMode = ChoiceCombination->GetSelection();



  return 1;
}

void SelectSegmentation::OnButtonOkClick(wxCommandEvent& event)
{
    saveSegmentationValuesFromForm();
    Close();
}

void SelectSegmentation::OnButtonCancelClick(wxCommandEvent& event)
{
    Close();
}

void SelectSegmentation::OnButtonExportClick(wxCommandEvent& event)
{
   FileDialogExport->ShowModal();


    char cstring[2048];
    strncpy(cstring, (const char*) FileDialogExport->GetPath().mb_str(wxConvUTF8), 2047);


    if (! saveSegmentationDataToFile(cstring, &selectedRGBConf , &selectedDepthConf ) )
    {
        wxMessageBox(wxT("Could not write Segmentation to the file specified"),wxT("While trying to save segmentation.."));
    }

}
