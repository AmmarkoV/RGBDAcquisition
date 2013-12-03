#include "SelectTarget.h"

#include "../acquisition/Acquisition.h"

#include <wx/msgdlg.h>

//(*InternalHeaders(SelectTarget)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(SelectTarget)
const long SelectTarget::ID_STATICBOX1 = wxNewId();
const long SelectTarget::ID_COMBOBOX1 = wxNewId();
const long SelectTarget::ID_STATICTEXT1 = wxNewId();
const long SelectTarget::ID_STATICTEXT2 = wxNewId();
const long SelectTarget::ID_TEXTCTRL1 = wxNewId();
const long SelectTarget::ID_BUTTON1 = wxNewId();
const long SelectTarget::ID_BUTTON2 = wxNewId();
//*)

BEGIN_EVENT_TABLE(SelectTarget,wxDialog)
	//(*EventTable(SelectTarget)
	//*)
END_EVENT_TABLE()

SelectTarget::SelectTarget(wxWindow* parent,wxWindowID id)
{
	//(*Initialize(SelectTarget)
	Create(parent, id, _("Target Selection for REcording"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(339,212));
	StaticBox1 = new wxStaticBox(this, ID_STATICBOX1, _("Target Selection"), wxPoint(16,16), wxSize(312,184), 0, _T("ID_STATICBOX1"));
	ComboBoxTarget = new wxComboBox(this, ID_COMBOBOX1, wxEmptyString, wxPoint(24,64), wxSize(288,29), 0, 0, 0, wxDefaultValidator, _T("ID_COMBOBOX1"));
	ComboBoxTarget->SetSelection( ComboBoxTarget->Append(_("Save to Files")) );
	ComboBoxTarget->Append(_("Stream to Network"));
	ComboBoxTarget->Append(_("No Output , dry run"));
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Please select the output for your recording"), wxPoint(24,40), wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	LabelForTargetPath = new wxStaticText(this, ID_STATICTEXT2, _("Target Path"), wxPoint(24,104), wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	TextCtrlTargetPath = new wxTextCtrl(this, ID_TEXTCTRL1, _("frames/YourDatasetName"), wxPoint(24,120), wxSize(288,27), 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
	ButtonRecord = new wxButton(this, ID_BUTTON1, _("Record"), wxPoint(24,152), wxSize(192,29), 0, wxDefaultValidator, _T("ID_BUTTON1"));
	ButtonCancel = new wxButton(this, ID_BUTTON2, _("Cancel"), wxPoint(228,152), wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));

	Connect(ID_COMBOBOX1,wxEVT_COMMAND_COMBOBOX_SELECTED,(wxObjectEventFunction)&SelectTarget::OnComboBoxTargetSelected);
	Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectTarget::OnButtonRecordClick);
	Connect(ID_BUTTON2,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&SelectTarget::OnButtonCancelClick);
	//*)

	recording=0;
}

SelectTarget::~SelectTarget()
{
	//(*Destroy(SelectTarget)
	//*)
}


void SelectTarget::OnComboBoxTargetSelected(wxCommandEvent& event)
{
    switch ( ComboBoxTarget->GetSelection() )
    {
     case 0 :
             TextCtrlTargetPath->Enable();
             LabelForTargetPath->SetLabel(wxT("Folder name for file output"));


             if (TextCtrlTargetPath->GetValue().IsEmpty())
             {
               TextCtrlTargetPath->SetValue(wxT("frames/YourDatasetName"));
             }

             break;
     case 1 :
             TextCtrlTargetPath->Enable();
             LabelForTargetPath->SetLabel(wxT("IP:PORT for the stream ( 0.0.0.0:8080 ) "));

             if (TextCtrlTargetPath->GetValue().IsEmpty())
             {
               TextCtrlTargetPath->SetValue(wxT("0.0.0.0:8080"));
             }
             break;
     default :
             LabelForTargetPath->SetLabel(wxT("Dry Run!"));
             TextCtrlTargetPath->Disable();
             TextCtrlTargetPath->Clear();
             break;
    };
}

void SelectTarget::OnButtonRecordClick(wxCommandEvent& event)
{
  char targetPath[1024];
  wxString target = TextCtrlTargetPath->GetValue();
  strcpy( targetPath , target.mb_str() );

    switch ( ComboBoxTarget->GetSelection() )
    {
     case 0 :
               acquisitionInitiateTargetForFrames(moduleID,devID,targetPath);
             break;
     case 1 :
               acquisitionInitiateTargetForFrames(moduleID,devID,targetPath);
             break;
     default :
               acquisitionInitiateTargetForFrames(moduleID,devID,"/dev/null");
             break;
    };

    recording=1;

    Close();
}

void SelectTarget::OnButtonCancelClick(wxCommandEvent& event)
{
    Close();
}
