#ifndef SELECTMODULE_H
#define SELECTMODULE_H

//(*Headers(SelectModule)
#include <wx/combobox.h>
#include <wx/dialog.h>
#include <wx/button.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
//*)

class SelectModule: public wxDialog
{
	public:

		SelectModule(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~SelectModule();
        void ReloadDevicesForSelectedModule();

		unsigned int moduleSelected;
		unsigned int deviceSelected;
		unsigned int widthSelected;
		unsigned int heightSelected;
		unsigned int fpsSelected;

		char deviceNameSelected[512];

        int selectionMade ;
		//(*Declarations(SelectModule)
		wxTextCtrl* TextCtrlFPS;
		wxTextCtrl* TextCtrlHeight;
		wxStaticText* StaticText1;
		wxButton* ButtonStartModule;
		wxStaticText* StaticText3;
		wxButton* ButtonCancel;
		wxComboBox* ComboBoxDevice;
		wxStaticText* StaticText4;
		wxStaticText* StaticText5;
		wxStaticText* StaticText2;
		wxStaticText* StaticText6;
		wxComboBox* ComboBoxModule;
		wxTextCtrl* TextCtrlWidth;
		//*)

	protected:

		//(*Identifiers(SelectModule)
		static const long ID_BUTTON1;
		static const long ID_STATICTEXT1;
		static const long ID_COMBOBOX1;
		static const long ID_STATICTEXT2;
		static const long ID_STATICTEXT3;
		static const long ID_TEXTCTRL2;
		static const long ID_STATICTEXT4;
		static const long ID_TEXTCTRL3;
		static const long ID_STATICTEXT5;
		static const long ID_TEXTCTRL4;
		static const long ID_STATICTEXT6;
		static const long ID_BUTTON2;
		static const long ID_COMBOBOX2;
		//*)

	private:

		//(*Handlers(SelectModule)
		void OnButtonStartModuleClick(wxCommandEvent& event);
		void OnButtonCancelClick(wxCommandEvent& event);
		void OnComboBoxDeviceSelected(wxCommandEvent& event);
		void OnComboBoxModuleSelected(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
