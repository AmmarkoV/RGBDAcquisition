#ifndef SELECTSEGMENTATION_H
#define SELECTSEGMENTATION_H

//(*Headers(SelectSegmentation)
#include <wx/spinctrl.h>
#include <wx/dialog.h>
#include <wx/button.h>
#include <wx/statbox.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/choice.h>
//*)

class SelectSegmentation: public wxDialog
{
	public:

		SelectSegmentation(wxWindow* parent,wxWindowID id=wxID_ANY);
		virtual ~SelectSegmentation();

		//(*Declarations(SelectSegmentation)
		wxSpinCtrl* minG;
		wxTextCtrl* TextCtrl3;
		wxTextCtrl* TextCtrl10;
		wxSpinCtrl* maxR;
		wxStaticText* StaticText13;
		wxStaticText* StaticText14;
		wxStaticText* StaticText15;
		wxStaticText* StaticText17;
		wxTextCtrl* TextCtrl9;
		wxTextCtrl* TextCtrl22;
		wxSpinCtrl* minB;
		wxSpinCtrl* minR;
		wxTextCtrl* TextCtrl20;
		wxTextCtrl* TextCtrl21;
		wxTextCtrl* TextCtrl18;
		wxStaticText* StaticText18;
		wxChoice* ChoiceCombination;
		wxStaticText* StaticText1;
		wxStaticText* StaticText10;
		wxStaticText* StaticText16;
		wxStaticBox* StaticBox2;
		wxTextCtrl* TextCtrl5;
		wxStaticText* StaticText3;
		wxSpinCtrl* maxG;
		wxButton* ButtonCancel;
		wxTextCtrl* TextCtrl12;
		wxButton* ButtonOk;
		wxStaticText* StaticText8;
		wxStaticText* StaticText12;
		wxTextCtrl* TextCtrl16;
		wxTextCtrl* TextCtrl6;
		wxStaticBox* StaticBox1;
		wxStaticText* StaticText7;
		wxSpinCtrl* SpinCtrl2;
		wxTextCtrl* TextCtrl23;
		wxTextCtrl* TextCtrl1;
		wxStaticText* StaticText4;
		wxStaticText* StaticText5;
		wxStaticText* StaticText2;
		wxTextCtrl* TextCtrl11;
		wxStaticText* StaticText6;
		wxSpinCtrl* SpinCtrl1;
		wxSpinCtrl* maxB;
		wxTextCtrl* TextCtrl15;
		wxTextCtrl* TextCtrl8;
		wxStaticText* StaticText19;
		wxTextCtrl* TextCtrl7;
		wxStaticText* StaticText9;
		wxTextCtrl* TextCtrl4;
		wxTextCtrl* TextCtrl2;
		wxTextCtrl* TextCtrl19;
		wxTextCtrl* TextCtrl13;
		wxStaticText* StaticText11;
		wxTextCtrl* TextCtrl17;
		wxTextCtrl* TextCtrl14;
		//*)

	protected:

		//(*Identifiers(SelectSegmentation)
		static const long ID_STATICBOX2;
		static const long ID_BUTTON1;
		static const long ID_BUTTON2;
		static const long ID_STATICBOX1;
		static const long ID_STATICTEXT1;
		static const long ID_SPINCTRL1;
		static const long ID_SPINCTRL2;
		static const long ID_SPINCTRL3;
		static const long ID_STATICTEXT2;
		static const long ID_SPINCTRL4;
		static const long ID_SPINCTRL5;
		static const long ID_SPINCTRL6;
		static const long ID_STATICTEXT3;
		static const long ID_STATICTEXT4;
		static const long ID_STATICTEXT5;
		static const long ID_STATICTEXT6;
		static const long ID_TEXTCTRL1;
		static const long ID_TEXTCTRL2;
		static const long ID_STATICTEXT7;
		static const long ID_TEXTCTRL3;
		static const long ID_TEXTCTRL4;
		static const long ID_STATICTEXT8;
		static const long ID_SPINCTRL7;
		static const long ID_STATICTEXT9;
		static const long ID_SPINCTRL8;
		static const long ID_STATICTEXT10;
		static const long ID_TEXTCTRL5;
		static const long ID_TEXTCTRL6;
		static const long ID_STATICTEXT11;
		static const long ID_TEXTCTRL7;
		static const long ID_TEXTCTRL8;
		static const long ID_STATICTEXT12;
		static const long ID_TEXTCTRL9;
		static const long ID_TEXTCTRL10;
		static const long ID_TEXTCTRL11;
		static const long ID_TEXTCTRL12;
		static const long ID_TEXTCTRL13;
		static const long ID_TEXTCTRL14;
		static const long ID_STATICTEXT13;
		static const long ID_STATICTEXT14;
		static const long ID_STATICTEXT15;
		static const long ID_STATICTEXT16;
		static const long ID_TEXTCTRL15;
		static const long ID_TEXTCTRL16;
		static const long ID_TEXTCTRL17;
		static const long ID_STATICTEXT17;
		static const long ID_TEXTCTRL18;
		static const long ID_TEXTCTRL19;
		static const long ID_TEXTCTRL20;
		static const long ID_TEXTCTRL21;
		static const long ID_TEXTCTRL22;
		static const long ID_TEXTCTRL23;
		static const long ID_STATICTEXT18;
		static const long ID_STATICTEXT19;
		static const long ID_CHOICE1;
		//*)

	private:

		//(*Handlers(SelectSegmentation)
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
