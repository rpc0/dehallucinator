function elementSupportsAttribute(element, attribute) {
    var test = document.createElement(element);
    if (attribute in test) {
      return true;
    } else {
      return false;
    }
  };

function setTextAreaPlaceholder( textarea_id ) {
    if (!elementSupportsAttribute('textarea', 'placeholder')) {
        // Fallback for browsers that don't support HTML5 placeholder attribute
        $(textarea_id)
            .data("originalText", $(textarea_id).text())
            .css("color", "#999")
            .focus(function() {
                var $el = $(this);
                if (this.value == $el.data("originalText")) {
                this.value = "";
                }
            })
            .blur(function() {
            if (this.value == "") {
                this.value = $(this).data("originalText");
            }
            });
        } else {
        // Browser does support HTML5 placeholder attribute, so use it.
        $(textarea_id)
            .attr("placeholder", $(textarea_id).text())
            .text("");
        }
}

function bar_color_select(color, d) {
  if (typeof d === "object") {
    // for data point
    if (d.value > 100) {
      return "#6d2976";
    } else {
      return "#404040";
    }
  }
}

function create_context_score_chart( div_id, data ) {
  // https://stackoverflow.com/questions/77967689/rotated-barchart-in-a-billboard-js
  cnt = 0
  data_arr = []
  grp_arr = []
  for (const doc of data.response.docs) {
    grp_arr.push(doc.metadata.source)
    row = [doc.metadata.source]
    row = row.concat( Array.from({length: cnt++}).fill(null))
    row.push(doc.metadata.similarity_score)
    data_arr.push(row)
  }

  var chart2 = bb.generate({
    data: {
      columns: data_arr,
      type: "bar",
      groups: [grp_arr],
      color: bar_color_select
    },
    legend: {
      show: false
    },
    axis: {
      rotated: true, 
      x: {
        type: "category",
              categories: grp_arr,
      },
      y: {
        label: "Value"
      }
    },
    bindto: div_id
  });

}

  