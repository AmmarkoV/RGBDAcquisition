#ifndef SELECTTARGET_H
#define SELECTTARGET_H

//(*Headers(SelectTarget)
#include <wx/combobox.h>
#include <wx/dialog.h>
#include <wx/button.h>
#include <wx/statbox.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
//*)

class SelectTarget: public wxDialog
{
	public:

		SelectTarget(wxWindow* parent,wxWindowID id=wxID_ANY);
		virtual ~SelectTarget();

		//(*Declarations(SelectTarget)
		wxTextCtrl* TextCtrlTargetPath;
		wxStaticText* StaticText1;
		wxButton* ButtonCancel;
		wxStaticBox* StaticBox1;
		wxStaticText* LabelForTargetPath;
		wxButton* ButtonRecord;
		wxComboBox* ComboBoxTarget;
		//*)

	protected:

		//(*Identifiers(SelectTarget)
		static const long ID_STATICBOX1;
		static const long ID_COMBOBOX1;
		static const long ID_STATICTEXT1;
		static const long ID_STATICTEXT2;
		static const long ID_TEXTCTRL1;
		static const long ID_BUTTON1;
		static const long ID_BUTTON2;
		//*)

	private:

		//(*Handlers(SelectTarget)
		void OnComboBoxTargetSelected(wxCommandEvent& event);
		void OnButtonRecordClick(wxCommandEvent& event);
		void OnButtonCancelClick(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
