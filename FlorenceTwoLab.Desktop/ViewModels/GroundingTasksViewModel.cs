using System;
using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using FlorenceTwoLab.Core;

namespace FlorenceTwoLab.Desktop.ViewModels;

public partial class GroundingTaskGroupViewModel : ObservableObject, ITaskGroupViewModel
{
    [ObservableProperty] private IReadOnlyCollection<Florence2TaskType> _predefinedTasks =
    [
        Florence2TaskType.CaptionToGrounding,
        Florence2TaskType.ReferringExpressionSegmentation,
        Florence2TaskType.OpenVocabularyDetection
    ];

    [ObservableProperty] private Florence2TaskType _selectedTask;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasCustomPrompt))]
    private string? _customPrompt;

    public GroundingTaskGroupViewModel()
    {
        SelectedTask = _predefinedTasks.First();
    }
    
    public string Header => "Grounding";

    public bool HasCustomPrompt => !String.IsNullOrWhiteSpace(CustomPrompt);

    public void SelectFirstTask()
    {
        SelectedTask = PredefinedTasks.First();
    }

    public Florence2Query Query() => Florence2Tasks.CreateQuery(SelectedTask, CustomPrompt ?? string.Empty);
}
