#ifndef ADDNEWELEMENT_H
#define ADDNEWELEMENT_H

//(*Headers(AddNewElement)
#include <wx/dialog.h>
#include <wx/button.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/choice.h>
//*)

class AddNewElement: public wxDialog
{
	public:

		AddNewElement(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~AddNewElement();

		//(*Declarations(AddNewElement)
		wxStaticText* StaticText1;
		wxButton* ButtonAdd;
		wxButton* ButtonCancel;
		wxChoice* Choice1;
		wxTextCtrl* TextCtrl1;
		wxStaticText* StaticText2;
		//*)

	protected:

		//(*Identifiers(AddNewElement)
		static const long ID_BUTTON1;
		static const long ID_STATICTEXT1;
		static const long ID_BUTTON2;
		static const long ID_TEXTCTRL1;
		static const long ID_STATICTEXT2;
		static const long ID_CHOICE1;
		//*)

	private:

		//(*Handlers(AddNewElement)
		void OnButtonCancelClick(wxCommandEvent& event);
		void OnButtonAddClick(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
