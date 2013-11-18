#ifndef SELECTSEGMENTATION_H
#define SELECTSEGMENTATION_H

//(*Headers(SelectSegmentation)
#include <wx/dialog.h>
#include <wx/button.h>
#include <wx/statbox.h>
//*)

class SelectSegmentation: public wxDialog
{
	public:

		SelectSegmentation(wxWindow* parent,wxWindowID id=wxID_ANY);
		virtual ~SelectSegmentation();

		//(*Declarations(SelectSegmentation)
		wxButton* Button1;
		wxButton* Button2;
		wxStaticBox* StaticBox2;
		wxStaticBox* StaticBox1;
		//*)

	protected:

		//(*Identifiers(SelectSegmentation)
		static const long ID_STATICBOX2;
		static const long ID_BUTTON1;
		static const long ID_BUTTON2;
		static const long ID_STATICBOX1;
		//*)

	private:

		//(*Handlers(SelectSegmentation)
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
