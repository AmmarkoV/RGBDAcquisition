#ifndef SELECTACQUISITIONGRAPH_H
#define SELECTACQUISITIONGRAPH_H

//(*Headers(SelectAcquisitionGraph)
#include <wx/combobox.h>
#include <wx/dialog.h>
#include <wx/panel.h>
//*)

class SelectAcquisitionGraph: public wxDialog
{
	public:

		SelectAcquisitionGraph(wxWindow* parent,wxWindowID id=wxID_ANY);
		virtual ~SelectAcquisitionGraph();

		//(*Declarations(SelectAcquisitionGraph)
		wxPanel* Panel1;
		wxComboBox* ComboBox1;
		//*)

	protected:

		//(*Identifiers(SelectAcquisitionGraph)
		static const long ID_COMBOBOX1;
		static const long ID_PANEL1;
		//*)

	private:

		//(*Handlers(SelectAcquisitionGraph)
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
