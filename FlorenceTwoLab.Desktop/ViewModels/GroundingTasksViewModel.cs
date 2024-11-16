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

    [ObservableProperty] private string? _customPrompt;

    private Action<Florence2TaskType>? _runTask;

    public GroundingTaskGroupViewModel()
    {
        SelectedTask = _predefinedTasks.First();
    }
    
    public string Header => "Grounding";

    public ITaskGroupViewModel Initialize(Action<Florence2TaskType> runTask)
    {
        _runTask = runTask;
        return this;
    }

    public void SelectFirstTask()
    {
        // we can not rely on the change event to trigger the task
        if (SelectedTask == PredefinedTasks.First()) _runTask?.Invoke(SelectedTask);
        SelectedTask = PredefinedTasks.First();
    }

    public Florence2Query Query() => Florence2Tasks.CreateQuery(SelectedTask, CustomPrompt ?? string.Empty);

    public void Run()
    {
        _runTask?.Invoke(SelectedTask);
    }

    partial void OnSelectedTaskChanged(Florence2TaskType value) => _runTask?.Invoke(value);
}