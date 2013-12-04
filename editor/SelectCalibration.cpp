#include "SelectCalibration.h"

#include <wx/msgdlg.h>
//(*InternalHeaders(SelectCalibration)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(SelectCalibration)
const long SelectCalibration::ID_STATICBOX1 = wxNewId();
const long SelectCalibration::ID_STATICTEXT1 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL1 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL2 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL3 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL4 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL5 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL6 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL7 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL8 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL9 = wxNewId();
const long SelectCalibration::ID_STATICTEXT2 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL10 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL11 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL12 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL13 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL14 = wxNewId();
const long SelectCalibration::ID_STATICTEXT3 = wxNewId();
const long SelectCalibration::ID_STATICTEXT4 = wxNewId();
const long SelectCalibration::ID_STATICTEXT5 = wxNewId();
const long SelectCalibration::ID_STATICTEXT6 = wxNewId();
const long SelectCalibration::ID_STATICTEXT7 = wxNewId();
const long SelectCalibration::ID_STATICTEXT8 = wxNewId();
const long SelectCalibration::ID_STATICBOX2 = wxNewId();
const long SelectCalibration::ID_STATICTEXT9 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL15 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL16 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL17 = wxNewId();
const long SelectCalibration::ID_STATICTEXT10 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL18 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL19 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL20 = wxNewId();
const long SelectCalibration::ID_STATICTEXT11 = wxNewId();
const long SelectCalibration::ID_STATICTEXT12 = wxNewId();
const long SelectCalibration::ID_STATICTEXT13 = wxNewId();
const long SelectCalibration::ID_STATICTEXT15 = wxNewId();
const long SelectCalibration::ID_STATICTEXT19 = wxNewId();
const long SelectCalibration::ID_COMBOBOX1 = wxNewId();
const long SelectCalibration::ID_BUTTON1 = wxNewId();
const long SelectCalibration::ID_BUTTON2 = wxNewId();
const long SelectCalibration::ID_STATICBOX3 = wxNewId();
const long SelectCalibration::ID_STATICTEXT20 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL21 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL22 = wxNewId();
const long SelectCalibration::ID_STATICTEXT21 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL23 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL24 = wxNewId();
const long SelectCalibration::ID_BUTTON3 = wxNewId();
const long SelectCalibration::ID_BUTTON4 = wxNewId();
const long SelectCalibration::ID_TEXTCTRL25 = wxNewId();
const long SelectCalibration::ID_CHECKBOX1 = wxNewId();
const long SelectCalibration::ID_CHECKBOX2 = wxNewId();
//*)

BEGIN_EVENT_TABLE(SelectCalibration,wxDialog)
	//(*EventTable(SelectCalibration)
	//*)
END_EVENT_TABLE()

SelectCalibration::SelectCalibration(wxWindow* parent,wxWindowID id)
{
	//(*Initialize(SelectCalibration)
	Create(parent, id, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(514,643));
	StaticBox1 = new wxStaticBox(this, ID_STATICBOX1, _("Intrinsics"), wxPoint(24,8), wxSize(464,216), 0, _T("ID_STATICBOX1"));
	fxLabel = new wxStaticText(this, ID_STATICTEXT1, _("Fx : 0.000000"), wxPoint(376,56), wxSize(94,17), 0, _T("ID_STATICTEXT1"));
	camera0 = new wxTextCtrl(this, ID_TEXTCTRL1, _("0.0"), wxPoint(56,56), wxSize(96,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
	camera1 = new wxTextCtrl(this, ID_TEXTCTRL2, _("0.0"), wxPoint(160,56), wxSize(96,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL2"));
	camera2 = new wxTextCtrl(this, ID_TEXTCTRL3, _("0.0"), wxPoint(264,56), wxSize(104,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL3"));
	camera3 = new wxTextCtrl(this, ID_TEXTCTRL4, _("0.0"), wxPoint(56,88), wxSize(96,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL4"));
	camera4 = new wxTextCtrl(this, ID_TEXTCTRL5, _("0.0"), wxPoint(160,88), wxSize(96,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL5"));
	camera5 = new wxTextCtrl(this, ID_TEXTCTRL6, _("0.0"), wxPoint(264,88), wxSize(104,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL6"));
	camera6 = new wxTextCtrl(this, ID_TEXTCTRL7, _("0.0"), wxPoint(56,120), wxSize(96,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL7"));
	camera7 = new wxTextCtrl(this, ID_TEXTCTRL8, _("0.0"), wxPoint(160,120), wxSize(96,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL8"));
	camera8 = new wxTextCtrl(this, ID_TEXTCTRL9, _("0.0"), wxPoint(264,120), wxSize(104,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL9"));
	StaticText2 = new wxStaticText(this, ID_STATICTEXT2, _("Distortion Coefficients"), wxPoint(40,152), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	k1 = new wxTextCtrl(this, ID_TEXTCTRL10, _("0.0"), wxPoint(32,184), wxSize(86,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL10"));
	k2 = new wxTextCtrl(this, ID_TEXTCTRL11, _("0.0"), wxPoint(120,184), wxSize(86,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL11"));
	p1 = new wxTextCtrl(this, ID_TEXTCTRL12, _("0.0"), wxPoint(208,184), wxSize(86,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL12"));
	p2 = new wxTextCtrl(this, ID_TEXTCTRL13, _("0.0"), wxPoint(296,184), wxSize(86,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL13"));
	k3 = new wxTextCtrl(this, ID_TEXTCTRL14, _("0.0"), wxPoint(384,184), wxSize(86,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL14"));
	StaticText3 = new wxStaticText(this, ID_STATICTEXT3, _("k1"), wxPoint(72,168), wxDefaultSize, 0, _T("ID_STATICTEXT3"));
	StaticText4 = new wxStaticText(this, ID_STATICTEXT4, _("k2"), wxPoint(160,168), wxDefaultSize, 0, _T("ID_STATICTEXT4"));
	StaticText5 = new wxStaticText(this, ID_STATICTEXT5, _("p1"), wxPoint(240,168), wxDefaultSize, 0, _T("ID_STATICTEXT5"));
	StaticText6 = new wxStaticText(this, ID_STATICTEXT6, _("p2"), wxPoint(328,168), wxDefaultSize, 0, _T("ID_STATICTEXT6"));
	StaticText7 = new wxStaticText(this, ID_STATICTEXT7, _("k3"), wxPoint(416,168), wxDefaultSize, 0, _T("ID_STATICTEXT7"));
	StaticText8 = new wxStaticText(this, ID_STATICTEXT8, _("Intrinsics Matrix"), wxPoint(40,32), wxDefaultSize, 0, _T("ID_STATICTEXT8"));
	StaticBox2 = new wxStaticBox(this, ID_STATICBOX2, _("Extrinsics"), wxPoint(24,232), wxSize(464,208), 0, _T("ID_STATICBOX2"));
	StaticText9 = new wxStaticText(this, ID_STATICTEXT9, _("Translation"), wxPoint(48,256), wxDefaultSize, 0, _T("ID_STATICTEXT9"));
	tX = new wxTextCtrl(this, ID_TEXTCTRL15, _("0.0"), wxPoint(48,280), wxSize(72,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL15"));
	tY = new wxTextCtrl(this, ID_TEXTCTRL16, _("0.0"), wxPoint(128,280), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL16"));
	tZ = new wxTextCtrl(this, ID_TEXTCTRL17, _("0.0"), wxPoint(200,280), wxSize(72,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL17"));
	StaticText10 = new wxStaticText(this, ID_STATICTEXT10, _("Rotation"), wxPoint(48,308), wxDefaultSize, 0, _T("ID_STATICTEXT10"));
	rX = new wxTextCtrl(this, ID_TEXTCTRL18, _("0.0"), wxPoint(48,328), wxSize(72,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL18"));
	rY = new wxTextCtrl(this, ID_TEXTCTRL19, _("0.0"), wxPoint(128,328), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL19"));
	rZ = new wxTextCtrl(this, ID_TEXTCTRL20, _("0.0"), wxPoint(200,328), wxSize(72,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL20"));
	fyLabel = new wxStaticText(this, ID_STATICTEXT11, _("Fy : 0.000000"), wxPoint(376,72), wxSize(88,17), 0, _T("ID_STATICTEXT11"));
	cxLabel = new wxStaticText(this, ID_STATICTEXT12, _("Cx : 0.000000"), wxPoint(376,112), wxSize(88,17), 0, _T("ID_STATICTEXT12"));
	cyLabel = new wxStaticText(this, ID_STATICTEXT13, _("Cy : 0.000000"), wxPoint(376,128), wxSize(88,17), 0, _T("ID_STATICTEXT13"));
	StaticText15 = new wxStaticText(this, ID_STATICTEXT15, _("Generated 4x4 Matrix"), wxPoint(280,256), wxDefaultSize, 0, _T("ID_STATICTEXT15"));
	StaticText19 = new wxStaticText(this, ID_STATICTEXT19, _("Scale"), wxPoint(48,372), wxDefaultSize, 0, _T("ID_STATICTEXT19"));
	ComboBoxScale = new wxComboBox(this, ID_COMBOBOX1, wxEmptyString, wxPoint(104,368), wxSize(168,25), 0, 0, 0, wxDefaultValidator, _T("ID_COMBOBOX1"));
	ComboBoxScale->SetSelection( ComboBoxScale->Append(_("meters")) );
	ComboBoxScale->Append(_("centimeters"));
	ComboBoxScale->Append(_("millimeters"));
	ComboBoxScale->Append(_("erroneous unit"));
	ButtonOk = new wxButton(this, ID_BUTTON1, _("Ok"), wxPoint(296,568), wxSize(101,27), 0, wxDefaultValidator, _T("ID_BUTTON1"));
	ButtonCancel = new wxButton(this, ID_BUTTON2, _("Cancel"), wxPoint(400,568), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));
	StaticBox3 = new wxStaticBox(this, ID_STATICBOX3, _("Camera Rendering"), wxPoint(24,448), wxSize(464,88), 0, _T("ID_STATICBOX3"));
	StaticText20 = new wxStaticText(this, ID_STATICTEXT20, _("Near/Far Plane for Rendering"), wxPoint(40,472), wxDefaultSize, 0, _T("ID_STATICTEXT20"));
	nearPlane = new wxTextCtrl(this, ID_TEXTCTRL21, _("0"), wxPoint(256,466), wxDefaultSize, 0, wxDefaultValidator, _T("ID_TEXTCTRL21"));
	farPlane = new wxTextCtrl(this, ID_TEXTCTRL22, _("0"), wxPoint(352,466), wxDefaultSize, 0, wxDefaultValidator, _T("ID_TEXTCTRL22"));
	StaticText21 = new wxStaticText(this, ID_STATICTEXT21, _("Width / Height"), wxPoint(40,496), wxDefaultSize, 0, _T("ID_STATICTEXT21"));
	width = new wxTextCtrl(this, ID_TEXTCTRL23, _("640"), wxPoint(256,490), wxDefaultSize, 0, wxDefaultValidator, _T("ID_TEXTCTRL23"));
	height = new wxTextCtrl(this, ID_TEXTCTRL24, _("480"), wxPoint(352,490), wxDefaultSize, 0, wxDefaultValidator, _T("ID_TEXTCTRL24"));
	ButtonSave = new wxButton(this, ID_BUTTON3, _("Save To File"), wxPoint(16,568), wxSize(120,27), 0, wxDefaultValidator, _T("ID_BUTTON3"));
	ButtonLoad = new wxButton(this, ID_BUTTON4, _("Load From File"), wxPoint(144,568), wxSize(120,27), 0, wxDefaultValidator, _T("ID_BUTTON4"));
	ModelViewMatrix = new wxTextCtrl(this, ID_TEXTCTRL25, _("0.000000  0.000000  0.000000  0.000000\n0.000000  0.000000  0.000000  0.000000\n0.000000  0.000000  0.000000  0.000000\n0.000000  0.000000  0.000000  0.000000"), wxPoint(288,280), wxSize(184,112), wxTE_MULTILINE|wxTE_READONLY|wxVSCROLL|wxHSCROLL, wxDefaultValidator, _T("ID_TEXTCTRL25"));
	CheckBoxIntrinsics = new wxCheckBox(this, ID_CHECKBOX1, _("Intrinsics Declared"), wxPoint(88,536), wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX1"));
	CheckBoxIntrinsics->SetValue(false);
	CheckBoxExtrinsics = new wxCheckBox(this, ID_CHECKBOX2, _("Extrinsics Declared"), wxPoint(248,536), wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX2"));
	CheckBoxExtrinsics->SetValue(false);
	FileDialogSave = new wxFileDialog(this, _("Please Select the output file"), wxEmptyString, wxEmptyString, _("*.calib"), wxFD_DEFAULT_STYLE|wxFD_SAVE, wxDefaultPosition, wxDefaultSize, _T("wxFileDialog"));
	FileDialogLoad = new wxFileDialog(this, _("Please select a calibration file"), wxEmptyString, wxEmptyString, _("*.calib"), wxFD_DEFAULT_STYLE|wxFD_OPEN|wxFD_FILE_MUST_EXIST, wxDefaultPosition, wxDefaultSize, _T("wxFileDialog"));

	Connect(ID_TEXTCTRL1,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::Oncamera0Text);
	Connect(ID_TEXTCTRL3,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::Oncamera2Text);
	Connect(ID_TEXTCTRL5,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::Oncamera4Text);
	Connect(ID_TEXTCTRL6,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::Oncamera5Text);
	Connect(ID_TEXTCTRL15,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::OntXText);
	Connect(ID_TEXTCTRL16,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::OntYText);
	Connect(ID_TEXTCTRL17,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::OntZText);
	Connect(ID_TEXTCTRL18,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::OnrXText);
	Connect(ID_TEXTCTRL19,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::OnrYText);
	Connect(ID_TEXTCTRL20,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&SelectCalibration::OnrZText);
	Connect(ID_COMBOBOX1,wxEVT_COMMAND_COMBOBOX_SELECTED,(wxObjectEventFunction)&SelectCalibration::OnComboBoxScaleSelected);
	Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectCalibration::OnButtonOkClick);
	Connect(ID_BUTTON2,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectCalibration::OnButtonCancelClick);
	Connect(ID_BUTTON3,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectCalibration::OnButtonSaveClick);
	Connect(ID_BUTTON4,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectCalibration::OnButtonLoadClick);
	//*)


   userLikesTheNewCalibration=0;
}

SelectCalibration::~SelectCalibration()
{
	//(*Destroy(SelectCalibration)
	//*)
}

double SelectCalibration::scaleSelectionToDepthUnit(unsigned int inputSelection)
{
    switch (inputSelection)
    {
        case 0 : return 1000.0; //meters so to do millimeters we need to *1000
        case 1 : return 10.0;   //centimeters so to do millimeters we need to *100
        case 2 : return 1.0;    //millimeters so to do millimeters we need to *1
    };

    return 1.0;
}


unsigned int SelectCalibration::depthUnitToScaleSelection(double depthUnit)
{
    if (depthUnit==1000.0) { return 0; } //meters so to do millimeters we need to *1000
    if (depthUnit==10.0)   { return 1; } //centimeters so to do millimeters we need to *100
    if (depthUnit==1.0)    { return 2; } //millimeters so to do millimeters we need to *1

    return 3;
}


int SelectCalibration::updateModelViewMatrixBox()
{
  double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(&calib);
  if (m!=0)
     {
       char buf[2048];
       sprintf(buf,"%0.6f %0.6f %0.6f %0.6f\n%0.6f %0.6f %0.6f %0.6f\n%0.6f %0.6f %0.6f %0.6f\n%0.6f %0.6f %0.6f %0.6f\n",
               m[0],m[1],m[2],m[3],
               m[4],m[5],m[6],m[7],
               m[8],m[9],m[10],m[11],
               m[12],m[13],m[14],m[15]
              );

       wxString mystring = wxString::FromUTF8(buf);
       ModelViewMatrix->SetValue(mystring);

       free(m);
       return 1;
     }
  return 0;
}




int SelectCalibration::reloadCalibrationFormFromValues()
{
  CheckBoxIntrinsics->SetValue( (calib.intrinsicParametersSet!=0) );
  CheckBoxExtrinsics->SetValue( (calib.extrinsicParametersSet!=0) );

  wxString val;
  val.Clear(); val.Printf(wxT("%0.6f"),calib.intrinsic[0]); camera0->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.intrinsic[1]); camera1->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.intrinsic[2]); camera2->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.intrinsic[3]); camera3->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.intrinsic[4]); camera4->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.intrinsic[5]); camera5->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.intrinsic[6]); camera6->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.intrinsic[7]); camera7->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.intrinsic[8]); camera8->SetValue(val);

  val.Clear(); val.Printf(wxT(" fx : %0.6f"),calib.intrinsic[0]);
  fxLabel->SetLabel(val);
  val.Clear(); val.Printf(wxT(" fy : %0.6f"),calib.intrinsic[4]);
  fyLabel->SetLabel(val);
  val.Clear(); val.Printf(wxT(" cx : %0.6f"),calib.intrinsic[2]);
  cxLabel->SetLabel(val);
  val.Clear(); val.Printf(wxT(" cy : %0.6f"),calib.intrinsic[5]);
  cyLabel->SetLabel(val);


  val.Clear(); val.Printf(wxT("%0.6f"),calib.k1); k1->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.k2); k2->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.p1); p1->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.p2); p2->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.k3); k3->SetValue(val);



  val.Clear(); val.Printf(wxT("%0.6f"),calib.extrinsicTranslation[0]); tX->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.extrinsicTranslation[1]); tY->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.extrinsicTranslation[2]); tZ->SetValue(val);


  val.Clear(); val.Printf(wxT("%0.6f"),calib.extrinsicRotationRodriguez[0]); rX->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.extrinsicRotationRodriguez[1]); rY->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.extrinsicRotationRodriguez[2]); rZ->SetValue(val);


  updateModelViewMatrixBox();

  val.Clear(); val.Printf(wxT("%0.6f"),calib.nearPlane); nearPlane->SetValue(val);
  val.Clear(); val.Printf(wxT("%0.6f"),calib.farPlane);  farPlane->SetValue(val);

  ComboBoxScale->SetSelection(depthUnitToScaleSelection(calib.depthUnit));

  Refresh();
}


int SelectCalibration::saveCalibrationValuesFromForm()
{
    double dValue;

    if (CheckBoxIntrinsics->IsChecked()) { calib.intrinsicParametersSet = 1;  } else
                                         { calib.intrinsicParametersSet = 0;  }

    if (CheckBoxExtrinsics->IsChecked()) { calib.extrinsicParametersSet = 1;  } else
                                         { calib.extrinsicParametersSet = 0;  }

    if (camera0->GetValue().ToDouble(&dValue)) {   calib.intrinsic[0] = dValue; }
    if (camera1->GetValue().ToDouble(&dValue)) {   calib.intrinsic[1] = dValue; }
    if (camera2->GetValue().ToDouble(&dValue)) {   calib.intrinsic[2] = dValue; }
    if (camera3->GetValue().ToDouble(&dValue)) {   calib.intrinsic[3] = dValue; }
    if (camera4->GetValue().ToDouble(&dValue)) {   calib.intrinsic[4] = dValue; }
    if (camera5->GetValue().ToDouble(&dValue)) {   calib.intrinsic[5] = dValue; }
    if (camera6->GetValue().ToDouble(&dValue)) {   calib.intrinsic[6] = dValue; }
    if (camera7->GetValue().ToDouble(&dValue)) {   calib.intrinsic[7] = dValue; }
    if (camera8->GetValue().ToDouble(&dValue)) {   calib.intrinsic[8] = dValue; }

    if (k1->GetValue().ToDouble(&dValue)) {   calib.k1 = dValue; }
    if (k2->GetValue().ToDouble(&dValue)) {   calib.k2 = dValue; }
    if (p1->GetValue().ToDouble(&dValue)) {   calib.p1 = dValue; }
    if (p2->GetValue().ToDouble(&dValue)) {   calib.p2 = dValue; }
    if (k3->GetValue().ToDouble(&dValue)) {   calib.k3 = dValue; }

    if (tX->GetValue().ToDouble(&dValue)) {   calib.extrinsicTranslation[0] = dValue; }
    if (tY->GetValue().ToDouble(&dValue)) {   calib.extrinsicTranslation[1] = dValue; }
    if (tZ->GetValue().ToDouble(&dValue)) {   calib.extrinsicTranslation[2] = dValue; }

    if (rX->GetValue().ToDouble(&dValue)) {   calib.extrinsicRotationRodriguez[0] = dValue; }
    if (rY->GetValue().ToDouble(&dValue)) {   calib.extrinsicRotationRodriguez[1] = dValue; }
    if (rZ->GetValue().ToDouble(&dValue)) {   calib.extrinsicRotationRodriguez[2] = dValue; }

    if (nearPlane->GetValue().ToDouble(&dValue)) {   calib.nearPlane = dValue; }
    if (farPlane->GetValue().ToDouble(&dValue)) {   calib.farPlane = dValue; }

    calib.depthUnit=scaleSelectionToDepthUnit( ComboBoxScale->GetSelection() );
    return 1;
}


void SelectCalibration::OnButtonOkClick(wxCommandEvent& event)
{
    userLikesTheNewCalibration=1;
    saveCalibrationValuesFromForm();
    Close();
}

void SelectCalibration::OnButtonCancelClick(wxCommandEvent& event)
{
    Close();
}

void SelectCalibration::OnButtonSaveClick(wxCommandEvent& event)
{
    FileDialogSave->ShowModal();

    // assuming you want UTF-8, change the wxConv* parameter as needed
    char cstring[2048];
    strncpy(cstring, (const char*) FileDialogSave->GetPath().mb_str(wxConvUTF8), 2047);

    fprintf(stderr,"Saving Calibration to %s \n",cstring);
    saveCalibrationValuesFromForm();
    if (! WriteCalibration((char*) cstring,&calib) )
    {
        wxString msg; msg<<wxT("Could not save calibration to file ")<<FileDialogSave->GetPath();
        wxMessageBox(msg,wxT("While Saving Calibration"));
    }
}

void SelectCalibration::OnButtonLoadClick(wxCommandEvent& event)
{
    FileDialogLoad->ShowModal();

    // assuming you want UTF-8, change the wxConv* parameter as needed
    char cstring[2048];
    strncpy(cstring, (const char*) FileDialogLoad->GetPath().mb_str(wxConvUTF8), 2047);

    fprintf(stderr,"Loading Calibration from %s \n",cstring);
    if (! ReadCalibration((char*) cstring,calib.boardWidth,calib.boardHeight,&calib) )
    {
        wxString msg; msg<<wxT("Could not load calibration to file ")<<FileDialogLoad->GetPath();
        wxMessageBox(msg,wxT("While Loading Calibration"));
    }
    reloadCalibrationFormFromValues();
}

void SelectCalibration::Oncamera0Text(wxCommandEvent& event)
{
  wxString val;
  val.Clear(); val<<wxT(" fx : ")<<camera0->GetValue();
  fxLabel->SetLabel(val);
}

void SelectCalibration::Oncamera2Text(wxCommandEvent& event)
{
  wxString val;
  val.Clear(); val<<wxT(" cx : ")<<camera2->GetValue();
  cxLabel->SetLabel(val);
}

void SelectCalibration::Oncamera4Text(wxCommandEvent& event)
{
  wxString val;
  val.Clear(); val<<wxT(" fy : ")<<camera4->GetValue();
  fyLabel->SetLabel(val);
}

void SelectCalibration::Oncamera5Text(wxCommandEvent& event)
{
  wxString val;
  val.Clear(); val<<wxT(" cy : ")<<camera5->GetValue();
  cyLabel->SetLabel(val);
}

void SelectCalibration::OntXText(wxCommandEvent& event)
{
    double dValue;
    if (tX->GetValue().ToDouble(&dValue)) {   calib.extrinsicTranslation[0] = dValue; }
  updateModelViewMatrixBox();
}

void SelectCalibration::OntYText(wxCommandEvent& event)
{
    double dValue;
    if (tY->GetValue().ToDouble(&dValue)) {   calib.extrinsicTranslation[1] = dValue; }
  updateModelViewMatrixBox();
}

void SelectCalibration::OntZText(wxCommandEvent& event)
{
    double dValue;
    if (tZ->GetValue().ToDouble(&dValue)) {   calib.extrinsicTranslation[2] = dValue; }
    updateModelViewMatrixBox();
}

void SelectCalibration::OnrXText(wxCommandEvent& event)
{
    double dValue;
    if (rX->GetValue().ToDouble(&dValue)) {   calib.extrinsicRotationRodriguez[0] = dValue; }
    updateModelViewMatrixBox();
}

void SelectCalibration::OnrYText(wxCommandEvent& event)
{
    double dValue;
    if (rY->GetValue().ToDouble(&dValue)) {   calib.extrinsicRotationRodriguez[1] = dValue; }
    updateModelViewMatrixBox();
}

void SelectCalibration::OnrZText(wxCommandEvent& event)
{
    double dValue;
    if (rZ->GetValue().ToDouble(&dValue)) {   calib.extrinsicRotationRodriguez[2] = dValue; }
    updateModelViewMatrixBox();
}

void SelectCalibration::OnComboBoxScaleSelected(wxCommandEvent& event)
{
    calib.depthUnit=scaleSelectionToDepthUnit( ComboBoxScale->GetSelection() );
    updateModelViewMatrixBox();
}
