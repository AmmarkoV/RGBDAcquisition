#include "SelectSegmentation.h"

//(*InternalHeaders(SelectSegmentation)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(SelectSegmentation)
const long SelectSegmentation::ID_STATICBOX2 = wxNewId();
const long SelectSegmentation::ID_BUTTON1 = wxNewId();
const long SelectSegmentation::ID_BUTTON2 = wxNewId();
const long SelectSegmentation::ID_STATICBOX1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(SelectSegmentation,wxDialog)
	//(*EventTable(SelectSegmentation)
	//*)
END_EVENT_TABLE()

SelectSegmentation::SelectSegmentation(wxWindow* parent,wxWindowID id)
{
	//(*Initialize(SelectSegmentation)
	Create(parent, id, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(748,561));
	StaticBox2 = new wxStaticBox(this, ID_STATICBOX2, _("Depth"), wxPoint(392,16), wxSize(336,424), 0, _T("ID_STATICBOX2"));
	Button1 = new wxButton(this, ID_BUTTON1, _("Label"), wxPoint(328,480), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON1"));
	Button2 = new wxButton(this, ID_BUTTON2, _("Label"), wxPoint(152,472), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));
	StaticBox1 = new wxStaticBox(this, ID_STATICBOX1, _("RGB"), wxPoint(16,16), wxSize(352,432), 0, _T("ID_STATICBOX1"));
	//*)
}

SelectSegmentation::~SelectSegmentation()
{
	//(*Destroy(SelectSegmentation)
	//*)
}

