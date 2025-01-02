using System;
using FlorenceTwoLab.Core;

namespace FlorenceTwoLab.Desktop.ViewModels;

public interface ITaskGroupViewModel
{
    string Header { get; }
    
    void SelectFirstTask();

    Florence2Query? Query();
}
