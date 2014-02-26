#include "AddNewElement.h"
#include <wx/msgdlg.h>

//(*InternalHeaders(AddNewElement)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(AddNewElement)
const long AddNewElement::ID_BUTTON1 = wxNewId();
const long AddNewElement::ID_STATICTEXT1 = wxNewId();
const long AddNewElement::ID_BUTTON2 = wxNewId();
const long AddNewElement::ID_TEXTCTRL1 = wxNewId();
const long AddNewElement::ID_STATICTEXT2 = wxNewId();
const long AddNewElement::ID_CHOICE1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(AddNewElement,wxDialog)
	//(*EventTable(AddNewElement)
	//*)
END_EVENT_TABLE()

AddNewElement::AddNewElement(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(AddNewElement)
	Create(parent, id, _("Add new element from points"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(400,471));
	Move(wxDefaultPosition);
	ButtonAdd = new wxButton(this, ID_BUTTON1, _("Add"), wxPoint(16,384), wxSize(248,29), 0, wxDefaultValidator, _T("ID_BUTTON1"));
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Label"), wxPoint(24,24), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	ButtonCancel = new wxButton(this, ID_BUTTON2, _("Cancel"), wxPoint(296,384), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));
	TextCtrl1 = new wxTextCtrl(this, ID_TEXTCTRL1, _("Text"), wxPoint(72,16), wxDefaultSize, 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
	StaticText2 = new wxStaticText(this, ID_STATICTEXT2, _("How To Add"), wxPoint(24,56), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	ChoiceHowToAdd = new wxChoice(this, ID_CHOICE1, wxPoint(24,80), wxSize(360,29), 0, 0, 0, wxDefaultValidator, _T("ID_CHOICE1"));
	ChoiceHowToAdd->Append(_("Flood Fill Point Segmentation"));
	ChoiceHowToAdd->Append(_("Plane Segmentation"));

	Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&AddNewElement::OnButtonAddClick);
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

void AddNewElement::OnButtonAddClick(wxCommandEvent& event)
{
  if ( (segDepth==0) || (segRGB==0) ) { wxMessageBox(wxT("Cannot add this element since accomodation for the settings is not allocated"),wxT("Error Adding Element")); return ;  }

  switch (ChoiceHowToAdd->GetSelection())
  {
   case 0 :  //Chosen to interpret selected points as flood fill sources

   break;
   case 1 :  //Chosen to interpret selected points as a plane
             segDepth->enablePlaneSegmentation=1;
             /*
             segDepth->p1[3];
             segDepth->p2[3];
             segDepth->p3[3];*/
   break;
   default :
     wxMessageBox(wxT("No Selection of how to add points"),wxT("Please select"));
     return ;
   break;
  }
}
