const $ = window.$

// adapted from https://stackoverflow.com/a/48078807/1217368
$(document).ready(function () {
  $('pre.highlight').each(function (i) {
    if (!$(this).parent().hasClass('no-copy-button') && (!$(this).parent().parent() || !$(this).parent().parent().hasClass('no-copy-button'))) {
      // create an id for the current code section
      var currentId = 'codeBlock' + i

      // find the code section and add the id to it
      var codeSection = $(this).find('code')
      codeSection.attr('id', currentId)

      // now create the button, setting the clipboard target to the id
      var btn = document.createElement('a')
      btn.setAttribute('type', 'btn')
      btn.setAttribute('class', 'badge badge-light btn-copy-code')
      btn.setAttribute('data-clipboard-target', '#' + currentId)
      btn.innerHTML = '<i class="fal fa-copy"></i> Copy'
      $(this).after(btn)
    }
  })

  new ClipboardJS('.btn-copy-code')
})
