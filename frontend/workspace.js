function showWorkspace(name){for(const workspace of document.querySelectorAll('main > .workspace'))workspace.hidden=workspace.id!==name+'-workspace';for(const tab of document.querySelectorAll('[data-tab]'))tab.setAttribute('aria-pressed',String(tab.dataset.tab===name));}
for(const button of document.querySelectorAll('[data-tab]'))button.onclick=()=>showWorkspace(button.dataset.tab);
