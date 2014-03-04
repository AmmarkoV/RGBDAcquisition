#include "AddNewElement.h"
#include "../acquisition/Acquisition.h"
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
const long AddNewElement::ID_LISTCTRL1 = wxNewId();
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
	ButtonAdd = new wxButton(this, ID_BUTTON1, _("Add"), wxPoint(16,360), wxSize(248,53), 0, wxDefaultValidator, _T("ID_BUTTON1"));
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Element Name : "), wxPoint(24,24), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	ButtonCancel = new wxButton(this, ID_BUTTON2, _("Cancel"), wxPoint(296,368), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));
	TextCtrl1 = new wxTextCtrl(this, ID_TEXTCTRL1, wxEmptyString, wxPoint(144,20), wxSize(232,27), 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
	StaticText2 = new wxStaticText(this, ID_STATICTEXT2, _("How To Interpret Point list : "), wxPoint(24,56), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	ChoiceHowToAdd = new wxChoice(this, ID_CHOICE1, wxPoint(24,80), wxSize(360,29), 0, 0, 0, wxDefaultValidator, _T("ID_CHOICE1"));
	ChoiceHowToAdd->Append(_("Flood Fill Point Segmentation"));
	ChoiceHowToAdd->Append(_("Plane Segmentation"));
	ListCtrlCopiedPointList = new wxListCtrl(this, ID_LISTCTRL1, wxPoint(24,112), wxSize(360,248), 0, wxDefaultValidator, _T("ID_LISTCTRL1"));

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

double getwxListDouble(wxListCtrl * theList , unsigned int col , unsigned int row )
{
  wxListItem     row_info;
  wxString       cell_contents_string;

  // Set what row it is (m_itemId is a member of the regular wxListCtrl class)
   row_info.m_itemId = row;
  // Set what column of that row we want to query for information.
   row_info.m_col = col;
  // Set text mask
   row_info.m_mask = wxLIST_MASK_TEXT;

   // Get the info and store it in row_info variable.
   theList->GetItem( row_info );

   // Extract the text out that cell
   cell_contents_string = row_info.m_text;

   double dValue;
    if (cell_contents_string.ToDouble(&dValue)) {  return dValue; }

   return 0.0;
}

unsigned int  getwxListInteger(wxListCtrl * theList , unsigned int col , unsigned int row )
{
  wxListItem     row_info;
  wxString       cell_contents_string;

  // Set what row it is (m_itemId is a member of the regular wxListCtrl class)
   row_info.m_itemId = row;
  // Set what column of that row we want to query for information.
   row_info.m_col = col;
  // Set text mask
   row_info.m_mask = wxLIST_MASK_TEXT;

   // Get the info and store it in row_info variable.
   theList->GetItem( row_info );

   // Extract the text out that cell
   cell_contents_string = row_info.m_text;

   unsigned long uValue;
    if (cell_contents_string.ToULong(&uValue)) {  return (unsigned int) uValue; }

   return 0;
}


void AddNewElement::OnButtonAddClick(wxCommandEvent& event)
{
  if ( (segDepth==0) || (segRGB==0) ) { wxMessageBox(wxT("Cannot add this element since accomodation for the settings is not allocated"),wxT("Error Adding Element")); return ;  }

  switch (ChoiceHowToAdd->GetSelection())
  {
   case 0 :  //Chosen to interpret selected points as flood fill sources

   break;
   case 1 :  //Chosen to interpret selected points as a plane

             if  ( ListCtrlPoints->GetItemCount() < 3 )
                      {  wxMessageBox(wxT("A plane needs at least 3 points"),wxT("Error Adding Element")); return ;  }

             segDepth->enablePlaneSegmentation=1;

             unsigned int x2D,y2D; float x,y,z;

             x2D=getwxListInteger(ListCtrlPoints,0,0); y2D=getwxListInteger(ListCtrlPoints,1,0);
             if ( acquisitionGetDepth3DPointAtXY(moduleID,devID,x2D,y2D,&x,&y,&z) )
                                    { segDepth->p1[0]=(double) x; segDepth->p1[1]=(double) y; segDepth->p1[2]=(double) z; } else
                                    { wxMessageBox(wxT("Could not project point 1 , it is left as it was"),wxT("Please select")); }


             x2D=getwxListInteger(ListCtrlPoints,0,1); y2D=getwxListInteger(ListCtrlPoints,1,1);
             if ( acquisitionGetDepth3DPointAtXY(moduleID,devID,x2D,y2D,&x,&y,&z) )
                                    { segDepth->p2[0]=(double) x; segDepth->p2[1]=(double) y; segDepth->p2[2]=(double) z; } else
                                    { wxMessageBox(wxT("Could not project point 2 , it is left as it was"),wxT("Please select")); }


             x2D=getwxListInteger(ListCtrlPoints,0,2); y2D=getwxListInteger(ListCtrlPoints,1,2);
             if ( acquisitionGetDepth3DPointAtXY(moduleID,devID,x2D,y2D,&x,&y,&z) )
                                    { segDepth->p3[0]=(double) x; segDepth->p3[1]=(double) y; segDepth->p3[2]=(double) z; } else
                                    { wxMessageBox(wxT("Could not project point 3 , it is left as it was"),wxT("Please select")); }



             ListCtrlPoints->DeleteItem(2);
             ListCtrlPoints->DeleteItem(1);
             ListCtrlPoints->DeleteItem(0);
   break;
   default :
     wxMessageBox(wxT("No Selection of how to add points"),wxT("Please select"));
     return ;
   break;
  }

  Close();
}
