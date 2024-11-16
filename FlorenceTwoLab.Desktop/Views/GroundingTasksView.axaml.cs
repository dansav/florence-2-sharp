using Avalonia.Controls;
using Avalonia.Input;
using FlorenceTwoLab.Desktop.ViewModels;

namespace FlorenceTwoLab.Desktop.Views;

public partial class GroundingTasksView : UserControl
{
    public GroundingTasksView()
    {
        InitializeComponent();
    }

    private void InputElement_OnKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && DataContext is GroundingTaskGroupViewModel vm)
        {
            vm.Run();
        }
    }
}