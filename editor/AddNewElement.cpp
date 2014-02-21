#include "AddNewElement.h"

//(*InternalHeaders(AddNewElement)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(AddNewElement)
const long AddNewElement::ID_BUTTON1 = wxNewId();
const long AddNewElement::ID_STATICTEXT1 = wxNewId();
const long AddNewElement::ID_BUTTON2 = wxNewId();
//*)

BEGIN_EVENT_TABLE(AddNewElement,wxDialog)
	//(*EventTable(AddNewElement)
	//*)
END_EVENT_TABLE()

AddNewElement::AddNewElement(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(AddNewElement)
	Create(parent, id, _("New Element"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxDefaultSize);
	Move(wxDefaultPosition);
	ButtonAdd = new wxButton(this, ID_BUTTON1, _("Add"), wxPoint(16,384), wxSize(248,29), 0, wxDefaultValidator, _T("ID_BUTTON1"));
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Label"), wxPoint(24,24), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	ButtonCancel = new wxButton(this, ID_BUTTON2, _("Cancel"), wxPoint(296,384), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));

	Connect(ID_BUTTON2,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&AddNewElement::OnButtonCancelClick);
	//*)
}

AddNewElement::~AddNewElement()
{
	//(*Destroy(AddNewElement)
	//*)
}


void AddNewElement::OnButtonCancelClick(wxCommandEvent& event)
{
    Close();
}
