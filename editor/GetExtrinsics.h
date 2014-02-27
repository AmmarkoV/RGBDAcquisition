#ifndef GETEXTRINSICS_H
#define GETEXTRINSICS_H

//(*Headers(GetExtrinsics)
#include <wx/spinctrl.h>
#include <wx/dialog.h>
#include <wx/button.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
//*)

class GetExtrinsics: public wxDialog
{
	public:

		unsigned int moduleID;
		unsigned int devID;

		GetExtrinsics(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~GetExtrinsics();

		//(*Declarations(GetExtrinsics)
		wxStaticText* StaticText1;
		wxStaticText* StaticText3;
		wxButton* ButtonGetExtrinsics;
		wxSpinCtrl* SpinCtrl2;
		wxTextCtrl* TextCtrl1;
		wxStaticText* StaticText4;
		wxStaticText* StaticText2;
		wxSpinCtrl* SpinCtrl1;
		//*)

	protected:

		//(*Identifiers(GetExtrinsics)
		static const long ID_STATICTEXT1;
		static const long ID_SPINCTRL1;
		static const long ID_STATICTEXT2;
		static const long ID_SPINCTRL2;
		static const long ID_STATICTEXT3;
		static const long ID_TEXTCTRL1;
		static const long ID_STATICTEXT4;
		static const long ID_BUTTON1;
		//*)

	private:

		//(*Handlers(GetExtrinsics)
		void OnButtonGetExtrinsicsClick(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
