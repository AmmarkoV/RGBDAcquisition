#include "ScanHuman.h"

//(*InternalHeaders(ScanHuman)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(ScanHuman)
const long ScanHuman::ID_GAUGE1 = wxNewId();
const long ScanHuman::ID_STATICBOX1 = wxNewId();
const long ScanHuman::ID_BUTTON1 = wxNewId();
const long ScanHuman::ID_TEXTCTRL1 = wxNewId();
const long ScanHuman::ID_BUTTON2 = wxNewId();
//*)

BEGIN_EVENT_TABLE(ScanHuman,wxDialog)
	//(*EventTable(ScanHuman)
	//*)
END_EVENT_TABLE()

ScanHuman::ScanHuman(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(ScanHuman)
	Create(parent, id, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(669,543));
	Move(wxDefaultPosition);
	Progress = new wxGauge(this, ID_GAUGE1, 100, wxPoint(24,496), wxSize(608,28), 0, wxDefaultValidator, _T("ID_GAUGE1"));
	StaticBox1 = new wxStaticBox(this, ID_STATICBOX1, _("Scan Human Dataset"), wxPoint(16,8), wxSize(632,520), 0, _T("ID_STATICBOX1"));
	ButtonCapture = new wxButton(this, ID_BUTTON1, _("Capture"), wxPoint(24,72), wxSize(128,30), 0, wxDefaultValidator, _T("ID_BUTTON1"));
	TextCtrlDataset = new wxTextCtrl(this, ID_TEXTCTRL1, wxEmptyString, wxPoint(24,40), wxSize(128,28), 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
	ButtonRestart = new wxButton(this, ID_BUTTON2, _("Restart"), wxPoint(24,104), wxSize(128,30), 0, wxDefaultValidator, _T("ID_BUTTON2"));
	//*)
}

ScanHuman::~ScanHuman()
{
	//(*Destroy(ScanHuman)
	//*)
}

