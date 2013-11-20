#include "SelectCalibration.h"

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
const long SelectCalibration::ID_STATICTEXT14 = wxNewId();
const long SelectCalibration::ID_STATICTEXT15 = wxNewId();
const long SelectCalibration::ID_STATICTEXT16 = wxNewId();
const long SelectCalibration::ID_STATICTEXT17 = wxNewId();
const long SelectCalibration::ID_STATICTEXT18 = wxNewId();
const long SelectCalibration::ID_STATICTEXT19 = wxNewId();
const long SelectCalibration::ID_COMBOBOX1 = wxNewId();
const long SelectCalibration::ID_BUTTON1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(SelectCalibration,wxDialog)
	//(*EventTable(SelectCalibration)
	//*)
END_EVENT_TABLE()

SelectCalibration::SelectCalibration(wxWindow* parent,wxWindowID id)
{
	//(*Initialize(SelectCalibration)
	Create(parent, id, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(501,564));
	StaticBox1 = new wxStaticBox(this, ID_STATICBOX1, _("Intrinsics"), wxPoint(24,8), wxSize(456,216), 0, _T("ID_STATICBOX1"));
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Fx : 0.0 "), wxPoint(288,56), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	TextCtrl1 = new wxTextCtrl(this, ID_TEXTCTRL1, _("0.0"), wxPoint(56,56), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
	TextCtrl2 = new wxTextCtrl(this, ID_TEXTCTRL2, _("0.0"), wxPoint(128,56), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL2"));
	TextCtrl3 = new wxTextCtrl(this, ID_TEXTCTRL3, _("0.0"), wxPoint(200,56), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL3"));
	TextCtrl4 = new wxTextCtrl(this, ID_TEXTCTRL4, _("0.0"), wxPoint(56,88), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL4"));
	TextCtrl5 = new wxTextCtrl(this, ID_TEXTCTRL5, _("0.0"), wxPoint(128,88), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL5"));
	TextCtrl6 = new wxTextCtrl(this, ID_TEXTCTRL6, _("0.0"), wxPoint(200,88), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL6"));
	TextCtrl7 = new wxTextCtrl(this, ID_TEXTCTRL7, _("0.0"), wxPoint(56,120), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL7"));
	TextCtrl8 = new wxTextCtrl(this, ID_TEXTCTRL8, _("0.0"), wxPoint(128,120), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL8"));
	TextCtrl9 = new wxTextCtrl(this, ID_TEXTCTRL9, _("0.0"), wxPoint(200,120), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL9"));
	StaticText2 = new wxStaticText(this, ID_STATICTEXT2, _("Distortion Coefficients"), wxPoint(40,152), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	TextCtrl10 = new wxTextCtrl(this, ID_TEXTCTRL10, _("0.0"), wxPoint(56,176), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL10"));
	TextCtrl11 = new wxTextCtrl(this, ID_TEXTCTRL11, _("0.0"), wxPoint(140,176), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL11"));
	TextCtrl12 = new wxTextCtrl(this, ID_TEXTCTRL12, _("0.0"), wxPoint(216,176), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL12"));
	TextCtrl13 = new wxTextCtrl(this, ID_TEXTCTRL13, _("0.0"), wxPoint(296,176), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL13"));
	TextCtrl14 = new wxTextCtrl(this, ID_TEXTCTRL14, _("0.0"), wxPoint(384,176), wxSize(48,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL14"));
	StaticText3 = new wxStaticText(this, ID_STATICTEXT3, _("k1"), wxPoint(32,178), wxDefaultSize, 0, _T("ID_STATICTEXT3"));
	StaticText4 = new wxStaticText(this, ID_STATICTEXT4, _("k2"), wxPoint(112,178), wxDefaultSize, 0, _T("ID_STATICTEXT4"));
	StaticText5 = new wxStaticText(this, ID_STATICTEXT5, _("p1"), wxPoint(192,176), wxDefaultSize, 0, _T("ID_STATICTEXT5"));
	StaticText6 = new wxStaticText(this, ID_STATICTEXT6, _("p2"), wxPoint(272,176), wxDefaultSize, 0, _T("ID_STATICTEXT6"));
	StaticText7 = new wxStaticText(this, ID_STATICTEXT7, _("k3"), wxPoint(352,176), wxDefaultSize, 0, _T("ID_STATICTEXT7"));
	StaticText8 = new wxStaticText(this, ID_STATICTEXT8, _("Intrinsics Matrix"), wxPoint(40,32), wxDefaultSize, 0, _T("ID_STATICTEXT8"));
	StaticBox2 = new wxStaticBox(this, ID_STATICBOX2, _("Extrinsics"), wxPoint(24,232), wxSize(456,208), 0, _T("ID_STATICBOX2"));
	StaticText9 = new wxStaticText(this, ID_STATICTEXT9, _("Translation"), wxPoint(48,256), wxDefaultSize, 0, _T("ID_STATICTEXT9"));
	TextCtrl15 = new wxTextCtrl(this, ID_TEXTCTRL15, _("0.0"), wxPoint(56,280), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL15"));
	TextCtrl16 = new wxTextCtrl(this, ID_TEXTCTRL16, _("0.0"), wxPoint(128,280), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL16"));
	TextCtrl17 = new wxTextCtrl(this, ID_TEXTCTRL17, _("0.0"), wxPoint(200,280), wxSize(72,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL17"));
	StaticText10 = new wxStaticText(this, ID_STATICTEXT10, _("Rotation"), wxPoint(48,308), wxDefaultSize, 0, _T("ID_STATICTEXT10"));
	TextCtrl18 = new wxTextCtrl(this, ID_TEXTCTRL18, _("0.0"), wxPoint(56,328), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL18"));
	TextCtrl19 = new wxTextCtrl(this, ID_TEXTCTRL19, _("0.0"), wxPoint(128,328), wxSize(64,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL19"));
	TextCtrl20 = new wxTextCtrl(this, ID_TEXTCTRL20, _("0.0"), wxPoint(200,328), wxSize(72,23), 0, wxDefaultValidator, _T("ID_TEXTCTRL20"));
	StaticText11 = new wxStaticText(this, ID_STATICTEXT11, _("Fy : 0.0"), wxPoint(288,72), wxDefaultSize, 0, _T("ID_STATICTEXT11"));
	StaticText12 = new wxStaticText(this, ID_STATICTEXT12, _("Cx : 0.0"), wxPoint(288,112), wxDefaultSize, 0, _T("ID_STATICTEXT12"));
	StaticText13 = new wxStaticText(this, ID_STATICTEXT13, _("Cy : 0.0"), wxPoint(288,128), wxDefaultSize, 0, _T("ID_STATICTEXT13"));
	StaticText14 = new wxStaticText(this, ID_STATICTEXT14, _("0.0     0.0      0.0      0.0"), wxPoint(304,272), wxDefaultSize, 0, _T("ID_STATICTEXT14"));
	StaticText15 = new wxStaticText(this, ID_STATICTEXT15, _("Generated 4x4 Matrix"), wxPoint(280,256), wxDefaultSize, 0, _T("ID_STATICTEXT15"));
	StaticText16 = new wxStaticText(this, ID_STATICTEXT16, _("0.0     0.0      0.0      0.0"), wxPoint(304,296), wxDefaultSize, 0, _T("ID_STATICTEXT16"));
	StaticText17 = new wxStaticText(this, ID_STATICTEXT17, _("0.0     0.0      0.0      0.0"), wxPoint(304,328), wxDefaultSize, 0, _T("ID_STATICTEXT17"));
	StaticText18 = new wxStaticText(this, ID_STATICTEXT18, _("0.0     0.0      0.0      0.0"), wxPoint(304,360), wxDefaultSize, 0, _T("ID_STATICTEXT18"));
	StaticText19 = new wxStaticText(this, ID_STATICTEXT19, _("Scale"), wxPoint(48,372), wxDefaultSize, 0, _T("ID_STATICTEXT19"));
	ComboBox1 = new wxComboBox(this, ID_COMBOBOX1, wxEmptyString, wxPoint(104,368), wxSize(168,25), 0, 0, 0, wxDefaultValidator, _T("ID_COMBOBOX1"));
	ComboBox1->Append(_("millimeters"));
	ComboBox1->Append(_("centimeters"));
	ComboBox1->SetSelection( ComboBox1->Append(_("meters")) );
	ButtonOk = new wxButton(this, ID_BUTTON1, _("Not Working Yet"), wxPoint(160,456), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON1"));

	Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectCalibration::OnButtonOkClick);
	//*)
}

SelectCalibration::~SelectCalibration()
{
	//(*Destroy(SelectCalibration)
	//*)
}


void SelectCalibration::OnButtonOkClick(wxCommandEvent& event)
{
    Close();
}
